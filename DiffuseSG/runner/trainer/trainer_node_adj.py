import logging
import os
import time

import torch
from torch import nn as nn
from torchvision.ops import box_iou, box_convert, complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss

from runner.sampler.sampler_node_adj import sg_go_sampling
from runner.trainer.trainer_utils import get_logger_per_epoch, update_epoch_learning_status, print_epoch_learning_status, \
    check_best_model, save_ckpt_model
from utils.arg_parser import set_training_loss_logger
from utils.dist_training import get_ddp_save_flag
from utils.attribute_code import attribute_converter
from utils.sg_utils import get_node_adj_num_type


def node_adj_move_forward_one_epoch(model, optimizer, ema_helper, dataloader, train_obj_gen, loss_func, epoch_logger,
                                    mode, dataset_name, node_encoding, edge_encoding, flag_sg,
                                    sanity_check_save_dir=None,
                                    flag_node_only=False, flag_binary_edge=False,
                                    iou_loss_type='iou', iou_loss_weight=0.0):
    """
    Go through one epoch of data. Compatible with training and testing.
    """
    assert mode in ['train', 'test']
    epoch_logger[mode]['time_start'] = time.time()
    # sanity_check_flag = epoch_logger['epoch'] == 0 and mode == 'train'
    assert node_encoding in ['one_hot', 'softmax', 'ddpm', 'bits']
    assert edge_encoding in ['one_hot', 'softmax', 'ddpm', 'bits']

    if flag_node_only or flag_binary_edge:
        assert flag_sg, 'flag_sg must be True if flag_node_only or flag_bin_edge is True'

    # one-hot encoding info
    info = get_node_adj_num_type(dataset_name, flag_sg, 'one_hot', flag_node_only, flag_node_bbox=False)
    oh_num_node_attr_type, oh_num_edge_attr_type = info['num_node_type'], info['num_adj_type']

    for data_tuple in dataloader:

        """Initialization"""
        if len(data_tuple) == 2:
            # adjs + nodes attributes
            adjs_gt, nodes_gt = data_tuple
            node_flags = torch.diagonal(adjs_gt, dim1=1, dim2=2) == -1  # [B, N] <- [B, N, N]
        elif len(data_tuple) == 3:
            # adjs + nodes + node flags
            adjs_gt, nodes_gt, node_flags = data_tuple
        elif len(data_tuple) == 4:
            # adjs + nodes + node flags + image_id_out
            adjs_gt, nodes_gt, node_flags, image_id_out = data_tuple
        else:
            raise NotImplementedError

        # enforce a large batch size, to stack the graphs multiple times
        if len(adjs_gt) < dataloader.batch_size and dataloader.batch_size % len(adjs_gt) == 0:
            if hasattr(dataloader, 'repeated_data'):
                adjs_gt, nodes_gt, node_flags = dataloader.repeated_data
            else:
                num_repeat = dataloader.batch_size // len(adjs_gt)
                adjs_gt = adjs_gt.repeat(num_repeat, *[1] * (adjs_gt.dim() - 1))
                nodes_gt = nodes_gt.repeat(num_repeat, *[1] * (nodes_gt.dim() - 1))
                node_flags = node_flags.repeat(num_repeat, *[1] * (node_flags.dim() - 1))
                repeated_data = [adjs_gt, nodes_gt, node_flags]
                dataloader.repeated_data = repeated_data

        adjs_gt = adjs_gt.to(train_obj_gen.dev)  # [B, N, N] or [B, C, N, N]
        nodes_gt = nodes_gt.to(train_obj_gen.dev)  # [B, N] or [B, N, C]
        node_flags = node_flags.to(train_obj_gen.dev)  # [B, N] or [B, N, N]

        # convert node and edge attributes to one-hot encoding if necessary
        if node_encoding == 'one_hot':
            if flag_sg and flag_node_only:
                # if flag_node_only, then nodes_gt is dummy and should not be converted
                nodes_gt = torch.zeros(adjs_gt.shape[:2], dtype=torch.float32, device=train_obj_gen.dev)
            elif flag_sg:
                assert nodes_gt.size(-1) == 5
                nodes_gt_type, nodes_gt_bbox = torch.split(nodes_gt, [1, 4], dim=-1)
                nodes_gt_type = nodes_gt_type.squeeze(-1)
                nodes_gt_type = attribute_converter(nodes_gt_type, node_flags, num_attr_type=oh_num_node_attr_type,
                                                    in_encoding='int', out_encoding='one_hot',
                                                    flag_nodes=True, flag_out_ddpm_range=True)  # [B, N, D]
                nodes_gt = torch.cat([nodes_gt_type, nodes_gt_bbox], dim=-1)  # [B, N, D + 4]
            else:
                nodes_gt = attribute_converter(nodes_gt, node_flags, num_attr_type=oh_num_node_attr_type,
                                               in_encoding='int', out_encoding='one_hot',
                                               flag_nodes=True, flag_out_ddpm_range=True)  # [B, N, D]

        if edge_encoding == 'one_hot':
            # [B, C, N, N] <- [B, N, N]
            adjs_gt = attribute_converter(adjs_gt, node_flags, num_attr_type=oh_num_edge_attr_type,
                                          in_encoding='int', out_encoding='one_hot',
                                          flag_adjs=True, flag_out_ddpm_range=True)  # [B, C, N, N]

        if train_obj_gen.objective == 'edm':
            net_input_a, net_input_x, net_cond, net_target_a, net_target_x, (
            c_skip, c_out, c_in, c_noise, sigmas, weights) = train_obj_gen.get_input_output(adjs_gt, nodes_gt, node_flags)
        else:
            raise NotImplementedError

        """Network forward pass"""
        if train_obj_gen.objective == 'edm':
            # Network forward pass
            def _edm_model_pass():
                # the model is with the precond module
                net_out_a, net_out_x = model(adjs=net_input_a, nodes=net_input_x, node_flags=node_flags, sigmas=sigmas)
                return net_out_a, net_out_x

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                net_output_a, net_output_x = _edm_model_pass()
            elif mode == 'test':
                with torch.no_grad():
                    net_output_a, net_output_x = _edm_model_pass()

            reg_loss_adj, reg_loss_node = loss_func(net_pred_a=net_output_a,
                                                    net_pred_x=net_output_x,
                                                    net_target_a=net_target_a,
                                                    net_target_x=net_target_x,
                                                    net_cond=net_cond,
                                                    adjs_perturbed=net_input_a,
                                                    adjs_gt=adjs_gt,
                                                    x_perturbed=net_input_x,
                                                    x_gt=nodes_gt,
                                                    node_flags=node_flags,
                                                    loss_weight=weights,
                                                    reduction='none')  # [B]
            
            # implemenet additional iou loss for bounding box
            if iou_loss_weight > 0.0:
                # convert to the range of [0, 1] from [-1, 1] linearly
                net_output_x_bbox = (net_output_x[..., -4:] + 1.0) / 2.0  # [B, N, 4]
                net_target_x_bbox = (net_target_x[..., -4:] + 1.0) / 2.0  # [B, N, 4]

                # convert to xyxy format
                net_output_x_bbox = box_convert(net_output_x_bbox, in_fmt='cxcywh', out_fmt='xyxy').clamp(min=0.0, max=1.0)
                net_target_x_bbox = box_convert(net_target_x_bbox, in_fmt='cxcywh', out_fmt='xyxy').clamp(min=0.0, max=1.0)
                if iou_loss_type == 'iou':
                    # compute vanilla IOU
                    bbox_iou_loss = box_iou(net_output_x_bbox.view(-1, 4), net_target_x_bbox.view(-1, 4))  # [B * N, B * N]
                    node_iou_loss = - (bbox_iou_loss.diag().view(-1)) ** 2.0  # [B * N]
                elif iou_loss_type == 'ciou':
                    # compute complete IOU
                    node_iou_loss = complete_box_iou_loss(net_output_x_bbox.view(-1, 4), net_target_x_bbox.view(-1, 4), reduction='none')  # [B * N]
                elif iou_loss_type == 'diou':
                    # compute distance IOU
                    node_iou_loss = distance_box_iou_loss(net_output_x_bbox.view(-1, 4), net_target_x_bbox.view(-1, 4), reduction='none')  # [B * N]
                elif iou_loss_type == 'giou' or iou_loss_type == 'giou_squared':
                    # compute generalized IOU
                    node_iou_loss = generalized_box_iou_loss(net_output_x_bbox.view(-1, 4), net_target_x_bbox.view(-1, 4), reduction='none')  # [B * N]
                    if iou_loss_type == 'giou_squared':
                        node_iou_loss = node_iou_loss ** 2.0
                else:
                    raise NotImplementedError
                node_flags_t = node_flags.view(-1)
                node_iou_loss = node_iou_loss * node_flags_t.to(torch.float32)  # [B * N, 1]
                node_iou_loss = node_iou_loss.view(-1, node_flags.shape[1])  # [B, N]
                node_iou_loss = node_iou_loss.sum(dim=-1) / node_flags_t.sum(dim=-1).to(torch.float32)  # [B]
                reg_loss_node = reg_loss_node + iou_loss_weight * node_iou_loss * weights
            if flag_node_only:
                # Sanity check
                # assert torch.equal(net_input_x, net_target_x)
                # assert torch.equal(net_input_x, net_output_x)
                # assert torch.equal(reg_loss_node, torch.zeros_like(reg_loss_node))
                reg_loss_node = reg_loss_node * 0.0

            loss = reg_loss_adj.mean() + reg_loss_node.mean()
        else:
            raise NotImplementedError

        """Network backward pass"""
        if mode == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)  # clip gradient
            optimizer.step()
            if ema_helper is not None:
                # we maintain a list EMA helper to handle multiple EMA coefficients
                [ema.update() for ema in ema_helper]

        """Record training result per iteration"""
        update_epoch_learning_status(epoch_logger, mode, reg_loss_adj=reg_loss_adj.clone().detach(),
                                     reg_loss_node=reg_loss_node.clone().detach(), noise_label=net_cond.detach())


def node_adj_go_training(model, optimizer, scheduler, ema_helper,
                         train_dl, test_dl, train_obj_gen, loss_func, mc_sampler, config, dist_helper, writer):
    """
    Core training functions go here.
    """

    """Initialization"""
    lowest_loss = {"epoch": -1, "loss": float('inf')}

    # Build txt loss file handler dedicated to training / evaluation loss per sample
    if get_ddp_save_flag():
        f_train_loss, f_test_loss = set_training_loss_logger(config.logdir)
    else:
        f_train_loss, f_test_loss = None, None

    save_interval = config.train.save_interval
    sample_interval = config.train.sample_interval
    sanity_check_save_dir = os.path.join(config.logdir, 'sanity_check_training_data')

    """scene graph dataset specific options"""
    flag_node_only = config.train.node_only
    flag_binary_edge = config.train.binary_edge
    if flag_node_only:
        logging.info("Node only generation mode is enabled for dataset {}".format(config.dataset.name))
    if flag_binary_edge:
        logging.info("Binary edge mode is enabled for dataset {}".format(config.dataset.name))

    node_encoding = config.train.node_encoding
    edge_encoding = config.train.edge_encoding
    iou_loss_type = config.train.iou_loss_type
    iou_loss_weight = config.train.iou_loss_weight

    """Go training"""
    for epoch in range(config.train.max_epoch):
        """Initialization"""
        epoch_logger = get_logger_per_epoch(epoch, flag_node_adj=True)
        if dist_helper.is_ddp:
            train_dl.sampler.set_epoch(epoch)
            test_dl.sampler.set_epoch(epoch)

        """Start learning"""
        # training
        model.train()
        train_dl.idx_to_word = test_dl.idx_to_word
        node_adj_move_forward_one_epoch(model, optimizer, ema_helper, train_dl, train_obj_gen, loss_func, epoch_logger,
                                        'train', config.dataset.name, node_encoding, edge_encoding,
                                        config.flag_sg, sanity_check_save_dir,
                                        flag_node_only, flag_binary_edge, iou_loss_type, iou_loss_weight)
        scheduler.step()
        logging.debug("epoch: {:05d}| effective learning rate: {:12.6f}".format(epoch, optimizer.param_groups[0]["lr"]))
        epoch_logger['lr'] = optimizer.param_groups[0]["lr"]

        # testing
        if epoch % save_interval == save_interval - 1 or epoch == 0:
            if ema_helper is not None:
                test_model = ema_helper[0].ema_model
            else:
                test_model = model
            test_model.eval()

            node_adj_move_forward_one_epoch(test_model, optimizer, ema_helper, test_dl, train_obj_gen, loss_func, epoch_logger,
                                            'test', config.dataset.name, node_encoding, edge_encoding,
                                            config.flag_sg, sanity_check_save_dir,
                                            flag_node_only, flag_binary_edge, iou_loss_type, iou_loss_weight)

            """Network weight saving"""
            # check best model
            check_best_model(model, ema_helper, epoch_logger, lowest_loss, save_interval, config, dist_helper)
            # save checkpoint model
            save_ckpt_model(model, ema_helper, epoch_logger, config, dist_helper)

        dist_helper.ddp_sync()

        # show the training and testing status
        print_epoch_learning_status(epoch_logger, f_train_loss, f_test_loss, writer, config.mcmc.name, flag_node_adj=True)

        """Sampling during training"""
        if ema_helper is not None:
            test_model = ema_helper[-1].ema_model
            ema_beta = ema_helper[-1].beta
        else:
            test_model = model
            ema_beta = 1.0
        test_model.eval()
        if epoch % sample_interval == 0:
            sampling_params = {'model_nm': 'training_e{:05d}'.format(epoch),
                               'weight_kw': '{:.3f}'.format(ema_beta),
                               'model_path': os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_{epoch:05d}.pth")}

            if config.flag_sg:
                pkl_data = test_dl.pkl_data
                idx_to_word = test_dl.idx_to_word
                triplet_to_count = test_dl.test_triplet_dict

                sg_go_sampling(epoch=epoch, model=model, dist_helper=dist_helper, eval_mode=False,
                               test_dl=test_dl, mc_sampler=mc_sampler, config=config, sanity_check=epoch == 0,
                               sampling_params=sampling_params,
                               triplet_to_count=triplet_to_count, flag_node_only=flag_node_only, flag_binary_edge=flag_binary_edge,
                               pkl_data=pkl_data, idx_to_word=idx_to_word,
                               writer=writer)

    # Destroy dedicated txt logger
    if get_ddp_save_flag():
        f_train_loss.close()
        f_test_loss.close()
