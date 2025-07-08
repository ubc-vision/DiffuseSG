import logging

import os
import time
import pandas as pd
import numpy as np

from evaluation.bbox_metrics import SceneGraphEvaluator


import torch
from torchvision.ops import box_convert

from utils.dist_training import get_ddp_save_flag
from utils.graph_utils import mask_adjs, mask_nodes
from utils.visual_utils import plot_scene_graph, plot_scene_graph_bbox
from utils.attribute_code import bin2dec, attribute_converter, reshape_node_attr_mat_to_vec
from utils.sg_utils import compute_sg_statistics, get_node_adj_num_type
from utils.dist_training import gather_tensors

from runner.sampler.sampler_utils import split_test_set


def sg_go_sampling(epoch, model, dist_helper, test_dl, mc_sampler, config,
                   sanity_check=False, eval_mode=False, sampling_params=None,
                   triplet_to_count=None, flag_node_only=False, flag_binary_edge=False,
                   pkl_data=None, idx_to_word=None, writer=None, skip_eval=False, random_node_num=False):
    """
    Create samples using the sampler and model.
    """

    """Initialization"""
    eval_size = config.test.eval_size
    flag_valid_eval_size = False
    if isinstance(eval_size, int):
        if eval_size > 0:
            flag_valid_eval_size = True

    num_nodes_total = [len(graph_dict['node_labels']) for graph_dict in test_dl.pkl_data] 

    if eval_mode:
        epoch_or_eval_stamp = 'eval_' + time.strftime('%b-%d-%H-%M-%S')
        shared_plot_dir = os.path.join(config.logdir, "sampling_during_evaluation")
        if flag_valid_eval_size:
            total_samples = eval_size
        else:
            total_samples = len(test_dl.dataset)
        batch_size = config.test.batch_size
    else:
        epoch_or_eval_stamp = 'eval_' + f"epoch_{epoch:05d}"
        shared_plot_dir = os.path.join(config.logdir, "sampling_during_training")
        if flag_valid_eval_size:
            total_samples = eval_size
        else:
            total_samples = config.train.batch_size
        batch_size = config.train.batch_size
    total_samples = min(len(test_dl.dataset), total_samples)  # cap the number of samples to generate
    os.makedirs(shared_plot_dir, exist_ok=True)
    logging.info("Sampling {:d} samples with batch size {:d}".format(total_samples, batch_size))

    flag_bbox = True
    node_encoding = config.train.node_encoding
    edge_encoding = config.train.edge_encoding
    flag_node_multi_channel = node_encoding != 'ddpm'
    flag_edge_multi_channel = edge_encoding != 'ddpm'

    assert node_encoding == edge_encoding
    info = get_node_adj_num_type(config.dataset.name, flag_sg=True, encoding=node_encoding, flag_node_only=flag_node_only, flag_node_bbox=flag_bbox)

    raw_num_node_type = info['raw_num_node_type']
    raw_num_adj_type = info['raw_num_adj_type']
    num_allowed_nodes = info['num_allowed_nodes']
    num_node_type = info['num_node_type']
    num_adj_type = info['num_adj_type']

    if flag_binary_edge:
        num_adj_type = 1
        flag_edge_multi_channel = False

    if flag_node_only:
        num_adj_type = num_node_type
        num_node_type = 5 if flag_bbox else 4
        flag_node_multi_channel = False

    # hyperparameter controlling the subset of interim adjs to store in memory
    max_num_interim_adjs = 10

    # Load testing data
    sampler_dl = split_test_set(test_dl, total_samples, batch_size, dist_helper, config.seed)

    """Draw samples and evaluate"""
    model.eval()

    """Draw samples from the MCMC sampler"""
    final_samples_a_ls, final_samples_x_ls, final_samples_node_flags_ls = [], [], []
    final_raw_a_ls, final_raw_x_ls = [], []
    final_samples_a_gt_ls, final_samples_x_gt_ls = [], []
    final_samples_bbox_ls, final_samples_bbox_gt_ls = [], []
    _sampler_dl_test_adjs_ls, _sampler_dl_test_nodes_ls, _sampler_dl_test_node_flags_ls = [], [], []
    _sampler_dl_test_image_id_ls = []
    i_generated = 0
    for i_iter, data_tuple in enumerate(sampler_dl):
        if len(data_tuple) == 3:
            test_adjs_gt, test_nodes_gt, test_node_flags = data_tuple
            test_image_id_out = -1
        elif len(data_tuple) == 4:
            # adjs + nodes + node flags + image_id_out
            test_adjs_gt, test_nodes_gt, test_node_flags, test_image_id_out = data_tuple
        else:
            raise ValueError("Invalid data_tuple length: {:d}".format(len(data_tuple)))
        test_adjs_gt = test_adjs_gt.to(config.dev)  # [B, N, N] or [B, C, N, N]
        test_nodes_gt = test_nodes_gt.to(config.dev)  # [B, N] or [B, N, C]
        test_node_flags = test_node_flags.to(config.dev)  # [B, N]

        # convert node and edge attributes to one-hot encoding if necessary
        if node_encoding == 'one_hot':
            if flag_node_only:
                # if flag_node_only, then nodes_gt is dummy and should not be converted
                # useful information is in adjs_gt
                test_nodes_gt = torch.zeros(test_adjs_gt.shape[:2], dtype=torch.float, device=config.dev)
            else:
                assert test_nodes_gt.size(-1) == 5
                test_nodes_gt_type, test_nodes_gt_bbox = torch.split(test_nodes_gt, [1, 4], dim=-1)
                test_nodes_gt_type = test_nodes_gt_type.squeeze(-1)
                test_nodes_gt_type = attribute_converter(test_nodes_gt_type, test_node_flags, num_attr_type=raw_num_node_type,
                                                    in_encoding='int', out_encoding='one_hot',
                                                    flag_nodes=True, flag_out_ddpm_range=True)  # [B, N, C]
                test_nodes_gt = torch.cat([test_nodes_gt_type, test_nodes_gt_bbox], dim=-1)

        if edge_encoding == 'one_hot':
            if flag_binary_edge:
                num_attr_type = 2
            elif flag_node_only:
                num_attr_type = raw_num_node_type  # adj is the original node attribute
            else:
                num_attr_type = raw_num_adj_type
            test_adjs_gt = attribute_converter(test_adjs_gt, test_node_flags, num_attr_type=num_attr_type,
                                                in_encoding='int', out_encoding='one_hot',
                                                flag_adjs=True, flag_out_ddpm_range=True)  # [B, C, N, N]

        # faithfully record whatever returned
        _sampler_dl_test_adjs_ls.append(test_adjs_gt)
        _sampler_dl_test_nodes_ls.append(test_nodes_gt)
        _sampler_dl_test_node_flags_ls.append(test_node_flags)
        _sampler_dl_test_image_id_ls.append(test_image_id_out)
        if random_node_num:            
            sample_num_nodes = np.random.choice(num_nodes_total, size=test_nodes_gt.size(0), replace=True)
            sample_node_flags = torch.ones_like(test_node_flags)
            for i in range(test_nodes_gt.size(0)):
                sample_node_flags[i, sample_num_nodes[i]:] = 0
            assert (sample_node_flags.sum(-1).gt(0.0) == test_node_flags.sum(-1).gt(0.0)).all()
        else:
            sample_node_flags = test_node_flags
        final_samples_node_flags_ls.append(sample_node_flags)

        # Initialize noisy data
        logging.info("--- Sampling from pure noisy data ---")
        init_adjs_sampler = None
        init_nodes_sampler = None

        logging.info("Generating [{:d} - {:d}]/ {:d} samples now... ({:d} / {:d} run)".format(
            i_generated, i_generated + test_adjs_gt.size(0), total_samples, i_iter + 1, len(sampler_dl)))
        i_generated += test_adjs_gt.size(0)

        # [B, N, N] + [T, B, N, N] + [B, N] + [T, B, N]
        final_samples_adjs, final_samples_nodes, interim_samples_adjs, interim_samples_nodes = mc_sampler.sample(
            model=model, node_flags=sample_node_flags,
            init_adjs=init_adjs_sampler, init_nodes=init_nodes_sampler,
            flag_interim_adjs=True,
            sanity_check_gt_adjs=test_adjs_gt if sanity_check else None,
            sanity_check_gt_nodes=test_nodes_gt if sanity_check else None,
            max_num_interim_adjs=max_num_interim_adjs,
            flag_node_multi_channel=flag_node_multi_channel,
            flag_adj_multi_channel=flag_edge_multi_channel,
            num_node_chan=num_node_type,
            num_edge_chan=num_adj_type,
        )

        def _node_only_gen_get_true_node_types(q_adj_tensor, node_flags, vector_size=None):
            """post-processing for node-only generation"""
            # q_adj_tensor: [B, N, N]; node types
            b, n = node_flags.shape[:2]
            out_q_node_flags = node_flags.view(b, -1)  # [B, N*N]
            if vector_size is None:
                _max_q_node = out_q_node_flags.sum(dim=1).max().item()  # int
            else:
                _max_q_node = vector_size

            out_q_adj_tensor = torch.zeros([b, _max_q_node, _max_q_node], device=q_adj_tensor.device)  # [B, M, M]
            out_q_node_tensor, out_q_node_flags = reshape_node_attr_mat_to_vec(q_adj_tensor, node_flags, vector_size=_max_q_node)
            return out_q_adj_tensor, out_q_node_tensor, out_q_node_flags

        # process bound box
        if flag_bbox:
            if flag_node_only:
                final_samples_adjs, final_samples_nodes_bbox = final_samples_adjs[:, :-4], final_samples_adjs[:, -4:]
                test_adjs_gt, test_nodes_bbox_gt = test_adjs_gt[:, :-4], test_adjs_gt[:, -4:]
            else:
                final_samples_nodes, final_samples_nodes_bbox = final_samples_nodes[..., :-4], final_samples_nodes[..., -4:]
                test_nodes_gt, test_nodes_bbox_gt = test_nodes_gt[..., :-4], test_nodes_gt[..., -4:]

            final_samples_nodes_bbox = final_samples_nodes_bbox * 0.5 + 0.5             # x y w h
            test_nodes_bbox_gt = test_nodes_bbox_gt * 0.5 + 0.5                         # x y w h

            if flag_node_only:
                final_samples_nodes_bbox = mask_adjs(final_samples_nodes_bbox.cpu(), sample_node_flags.cpu())
                test_nodes_bbox_gt = mask_adjs(test_nodes_bbox_gt.cpu(), test_node_flags.cpu())
            else:
                final_samples_nodes_bbox = mask_nodes(final_samples_nodes_bbox.cpu(), sample_node_flags.cpu())
                test_nodes_bbox_gt = mask_nodes(test_nodes_bbox_gt.cpu(), test_node_flags.cpu())

            final_samples_nodes_bbox_gt = test_nodes_bbox_gt

            if flag_node_only:
                # turn adjs into node bbox attributes
                # [B, 4, N, N] -> [B, X, 4]
                final_samples_nodes_bbox, _ = reshape_node_attr_mat_to_vec(final_samples_nodes_bbox.cpu(), sample_node_flags.cpu(), vector_size=num_allowed_nodes)
                test_nodes_bbox_gt, _ = reshape_node_attr_mat_to_vec(test_nodes_bbox_gt.cpu(), test_node_flags.cpu(), vector_size=num_allowed_nodes)
                final_samples_nodes_bbox_gt = test_nodes_bbox_gt

        """quantization based on different encoding methods"""
        def _decode_node(node_samples, node_flags, encoding_method):
            node_samples = node_samples.clamp(-1.0, 1.0)
            if encoding_method in ['bits', 'one_hot']:
                node_samples = torch.where(node_samples > 0.0, torch.ones_like(node_samples), -torch.ones_like(node_samples))
                node_samples = mask_nodes(node_samples, node_flags)
            if encoding_method == 'bits':
                # use explicit decoding with clamping to avoid numerical errors
                _q_binary_node = node_samples.gt(0.0).cpu().float()  # [B, N, C], -1/1 -> 0/1
                _q_binary_node = mask_nodes(_q_binary_node, node_flags.cpu())  # [B, N]
                _q_node = bin2dec(_q_binary_node, num_bits=np.ceil(np.log2(raw_num_node_type)).astype(int))  # [B, N]
                _q_node = mask_nodes(_q_node, node_flags.cpu()).clamp(min=0, max=raw_num_node_type-1)  # [B, N]
            else:
                if len(node_samples.shape) == 3 and node_samples.shape[-1] == 1:
                    node_samples = node_samples.squeeze(-1)
                _q_node = attribute_converter(in_attr=node_samples, attr_flags=node_flags.cpu(),
                                                in_encoding=encoding_method, out_encoding='int', num_attr_type=raw_num_node_type,
                                                flag_nodes=True, flag_adjs=False,
                                                flag_in_ddpm_range=True, flag_out_ddpm_range=False)
            return _q_node

        def _decode_adj(adj_samples, node_flags, encoding_method):
            adj_samples = adj_samples.clamp(-1.0, 1.0)
            if encoding_method in ['bits', 'one_hot']:
                adj_samples = torch.where(adj_samples > 0.0, torch.ones_like(adj_samples),
                                            -torch.ones_like(adj_samples))
                adj_samples = mask_adjs(adj_samples, node_flags)

            if encoding_method in ['ddpm', 'one_hot']:
                if encoding_method == 'ddpm':
                    _num_attr_type = raw_num_adj_type
                    if flag_node_only:
                        _num_attr_type = raw_num_node_type  # actually node type
                    if flag_binary_edge:
                        _num_attr_type = 2  # binary edge
                elif encoding_method == 'one_hot':
                    _num_attr_type = raw_num_adj_type
                else:
                    raise NotImplementedError
                _q_adj = attribute_converter(in_attr=adj_samples, attr_flags=node_flags.cpu(),
                                                in_encoding=encoding_method, out_encoding='int',
                                                num_attr_type=_num_attr_type,
                                                flag_nodes=True, flag_adjs=False,
                                                flag_in_ddpm_range=True, flag_out_ddpm_range=False)
            elif encoding_method == 'bits':
                if flag_binary_edge:
                    adj_samples = adj_samples.unsqueeze(1)  # [B, 1, N, N] <- [B, N, N]

                _q_binary_adj = adj_samples.gt(0.0).cpu().float()  # [B, C, N, N]
                _q_binary_adj = mask_adjs(_q_binary_adj, node_flags.cpu())  # [B, C, N, N]
                _q_binary_adj = _q_binary_adj.permute(0, 2, 3, 1)  # [B, N, N, C]
                if flag_node_only:
                    _q_adj = bin2dec(_q_binary_adj, num_bits=np.ceil(np.log2(raw_num_node_type)).astype(int))  # [B, N, N]
                    _q_adj = mask_adjs(_q_adj, node_flags.cpu()).clamp(min=0, max=raw_num_node_type-1)  # [B, N, N]
                else:
                    _q_adj = bin2dec(_q_binary_adj, num_bits=np.ceil(np.log2(raw_num_adj_type)).astype(int))  # [B, N, N]
                    _q_adj = mask_adjs(_q_adj, node_flags.cpu()).clamp(min=0, max=raw_num_adj_type-1)  # [B, N, N]
            else:
                raise NotImplementedError

            b, n = node_flags.shape[:2]
            if not flag_node_only:
                _q_adj[:, torch.eye(n, device=_q_adj.device).bool()] = 0.0  # [B, N, N]  # remove self-loops

            return _q_adj.contiguous()

        if flag_node_only:
            q_node, q_node_gt = None, None
        else:
            q_node = _decode_node(final_samples_nodes.cpu(), sample_node_flags.cpu(), node_encoding)
            q_node_gt = _decode_node(test_nodes_gt.cpu(), test_node_flags.cpu(), node_encoding)
        q_adj = _decode_adj(final_samples_adjs.cpu(), sample_node_flags.cpu(), edge_encoding)
        q_adj_gt = _decode_adj(test_adjs_gt.cpu(), test_node_flags.cpu(), edge_encoding)

        if flag_node_only:
            # turn adjs into node types
            # Warning: be careful at this step.
            # it may cause error in the DDP mode, as the effective tensor size in different GPUs may be different.
            q_adj, q_node, _ = _node_only_gen_get_true_node_types(q_adj.contiguous(), sample_node_flags.cpu(), num_allowed_nodes)
            q_adj_gt, q_node_gt, _ = _node_only_gen_get_true_node_types(q_adj_gt.contiguous(), test_node_flags.cpu(), num_allowed_nodes)

        final_raw_a_ls.append(final_samples_adjs.cpu())  # [B, N, N], before quantization
        final_raw_x_ls.append(final_samples_nodes.cpu())  # [B, N], before quantization
        final_samples_a_ls.append(q_adj.cpu())  # [B, N, N], quantized!
        final_samples_x_ls.append(q_node.cpu())  # [B, N], quantized!
        final_samples_a_gt_ls.append(q_adj_gt.cpu())  # [B, N, N], quantized!
        final_samples_x_gt_ls.append(q_node_gt.cpu())  # [B, N], quantized!

        if flag_bbox:
            final_samples_bbox_ls.append(final_samples_nodes_bbox.cpu())        # [B, N, 4]
            final_samples_bbox_gt_ls.append(final_samples_nodes_bbox_gt.cpu())  # [B, N, 4]

    # end of sample_dl loop
    final_raw_adjs = torch.cat(final_raw_a_ls, dim=0)                       # [B, N, N]
    final_raw_nodes = torch.cat(final_raw_x_ls, dim=0)                      # [B, N]
    final_samples_adjs = torch.cat(final_samples_a_ls, dim=0)               # [B, N, N]
    final_samples_nodes = torch.cat(final_samples_x_ls, dim=0)              # [B, N]
    final_samples_node_flags = torch.cat(final_samples_node_flags_ls, dim=0)  # [B, N]
    final_samples_adjs_gt = torch.cat(final_samples_a_gt_ls, dim=0)         # [B, N, N]
    final_samples_nodes_gt = torch.cat(final_samples_x_gt_ls, dim=0)        # [B, N]

    if flag_bbox:
        final_samples_nodes_bbox = torch.cat(final_samples_bbox_ls, dim=0)          # [B, N, 4]
        final_samples_nodes_bbox_gt = torch.cat(final_samples_bbox_gt_ls, dim=0)    # [B, N, 4]

    _sampler_dl_test_adjs = torch.cat(_sampler_dl_test_adjs_ls, dim=0).cpu()                # [B, N, N]
    _sampler_dl_test_nodes = torch.cat(_sampler_dl_test_nodes_ls, dim=0).cpu()                  # [B, N]
    _sampler_dl_test_node_flags = torch.cat(_sampler_dl_test_node_flags_ls, dim=0).cpu()    # [B, N]
    _sampler_dl_test_image_ids = torch.cat(_sampler_dl_test_image_id_ls, dim=0).cpu()       # [B]

    if dist_helper.is_ddp:
        final_raw_adjs = gather_tensors(final_raw_adjs, cat_dim=0, device=config.dev).cpu()
        final_raw_nodes = gather_tensors(final_raw_nodes, cat_dim=0, device=config.dev).cpu()
        final_samples_adjs = gather_tensors(final_samples_adjs, cat_dim=0, device=config.dev).cpu()
        final_samples_nodes = gather_tensors(final_samples_nodes, cat_dim=0, device=config.dev).cpu()
        final_samples_node_flags = gather_tensors(final_samples_node_flags, cat_dim=0, device=config.dev).cpu()
        final_samples_adjs_gt = gather_tensors(final_samples_adjs_gt, cat_dim=0, device=config.dev).cpu()
        final_samples_nodes_gt = gather_tensors(final_samples_nodes_gt, cat_dim=0, device=config.dev).cpu()
        _sampler_dl_test_adjs = gather_tensors(_sampler_dl_test_adjs, cat_dim=0, device=config.dev).cpu()
        _sampler_dl_test_nodes = gather_tensors(_sampler_dl_test_nodes, cat_dim=0, device=config.dev).cpu()
        _sampler_dl_test_node_flags = gather_tensors(_sampler_dl_test_node_flags, cat_dim=0, device=config.dev).cpu()
        _sampler_dl_test_image_ids = gather_tensors(_sampler_dl_test_image_ids, cat_dim=0, device=config.dev).cpu()
        if flag_bbox:
            final_samples_nodes_bbox = gather_tensors(final_samples_nodes_bbox, cat_dim=0, device=config.dev).cpu()
            final_samples_nodes_bbox_gt = gather_tensors(final_samples_nodes_bbox_gt, cat_dim=0, device=config.dev).cpu()

    """Compute MMD and visualize the final sample"""
    logging.info("Sampling and post-processing done.")
    if skip_eval:
        logging.info("Evaluataion will be skipped. Results are saved to disk.")
    else:
        logging.info("Now computing MMD and visualizing the final sample...")
    if get_ddp_save_flag():
        # Init
        model_signature = "{:s}".format(sampling_params['weight_kw'])
        plot_subdir = "{:s}_exp_{:s}_{:s}".format("pure_noise",
                                                    epoch_or_eval_stamp,
                                                    'sanity_check' if sanity_check else 'model_inference')
        plot_subdir = '_'.join([model_signature, plot_subdir])

        if sanity_check:
            plot_subtitle = "sanity_check"
        else:
            plot_subtitle = "pure_noise"

        if sampling_params is not None:
            fig_keyword = sampling_params['model_nm'] + '_weight_{:s}'.format(sampling_params['weight_kw'])
            plot_subdir = fig_keyword + '_' + plot_subdir
            plot_subtitle = fig_keyword + '_' + plot_subtitle
        fig_title = '{:s}_{:s}.png'.format(epoch_or_eval_stamp, plot_subtitle)
        sg_plot_path = os.path.join(shared_plot_dir, plot_subdir)
        path_plot_subdir = os.path.join(shared_plot_dir, plot_subdir)
        save_path_sg_txt = os.path.join(path_plot_subdir, 'gen_scene_graph.txt')
        path_final_samples_array = os.path.join(path_plot_subdir, 'final_samples_array.npz')
        os.makedirs(path_plot_subdir, exist_ok=True)

        # Note we must use exactly what is returned in the sampler_dl.
        # Otherwise, the node flags would be problematic and the final output would be wrong.
        test_adjs_gt = _sampler_dl_test_adjs.cpu()
        test_nodes_gt = _sampler_dl_test_nodes.cpu()
        test_node_flags = _sampler_dl_test_node_flags.cpu()
        test_image_ids = _sampler_dl_test_image_ids.cpu()
        final_samples_node_flags = final_samples_node_flags.cpu()

        if flag_node_only:
            _, _, test_node_flags = _node_only_gen_get_true_node_types(test_adjs_gt.contiguous(), test_node_flags, num_allowed_nodes)

        # visualize scene graphs in networkx
        plot_scene_graph(final_samples_nodes, final_samples_adjs, final_samples_node_flags, idx_to_word,
                            save_dir=sg_plot_path, title=fig_title, flag_bin_edge=flag_binary_edge, num_plots=8)
        
        # save the final samples withtout evaluation results
        # test_set_gt['gt_image_id'].append(gt_image_id[i].item())

        np.savez_compressed(os.path.join(path_plot_subdir, 'final_samples_array_before_eval.npz'),
                            samples_node_flags=final_samples_node_flags.cpu().bool().numpy(),
                            samples_a=final_samples_adjs.cpu().numpy(),
                            samples_x=final_samples_nodes.cpu().numpy(),
                            raw_a=final_raw_adjs.cpu().numpy(),
                            raw_x=final_raw_nodes.cpu().numpy(),
                            gt_node_flags=test_node_flags.cpu().bool().numpy(),
                            gt_a=final_samples_adjs_gt.cpu().numpy(),
                            gt_x=final_samples_nodes_gt.cpu().numpy(),
                            samples_x_bbox=final_samples_nodes_bbox.cpu().numpy() if flag_bbox else None,
                            gt_x_bbox=final_samples_nodes_bbox_gt.cpu().numpy() if flag_bbox else None,
                            gt_image_ids=test_image_ids.cpu().numpy(),
                            )

        if skip_eval:
            # return early
            return

        # evaluation
        logging.info(f'Number of generated scene graphs: {len(final_samples_node_flags)}')
        logging.info('=' * 100)

        # evaluate statistics
        if pkl_data is not None:
            result_data = {
                'samples_node_flags': final_samples_node_flags.cpu().bool().numpy(),
                'samples_a': final_samples_adjs.cpu().numpy(),
                'samples_x': final_samples_nodes.cpu().numpy(),
                'raw_a': final_raw_adjs.cpu().numpy(),
                'raw_x': final_raw_nodes.cpu().numpy(),
                'gt_node_flags': test_node_flags.cpu().bool().numpy(),
                'gt_a': final_samples_adjs_gt.cpu().numpy(),
                'gt_x': final_samples_nodes_gt.cpu().numpy(),
            }
            if flag_bbox:
                result_data['samples_x_bbox'] = final_samples_nodes_bbox.cpu().numpy()
                result_data['gt_x_bbox'] = final_samples_nodes_bbox_gt.cpu().numpy()
            if flag_binary_edge: 
                for i in range(len(pkl_data)):
                    pkl_data[i]['edge_map'] = np.where(pkl_data[i]['edge_map'] > 0, 1, 0)
            compute_sg_statistics(result_data, pkl_data, idx_to_word, os.path.join(shared_plot_dir, plot_subdir))

        # select the first non-repeated samples if total_samples > len(test_dl.dataset)
        if total_samples > len(test_dl.dataset):
            test_node_flags_gt = test_node_flags[:len(test_dl.dataset)]
            final_samples_nodes_gt = final_samples_nodes_gt[:len(test_dl.dataset)]
            final_samples_adjs_gt = final_samples_adjs_gt[:len(test_dl.dataset)]
        else:
            test_node_flags_gt = test_node_flags

        eval_helper = SceneGraphEvaluator()
        mmd_kernels = ['gaussian']
        node_deg_mmd = eval_helper.compute_node_degree_mmd(final_samples_adjs, final_samples_adjs_gt, mmd_kernels)
        node_type_mmd = eval_helper.compute_node_type_mmd(final_samples_nodes, final_samples_nodes_gt, final_samples_node_flags, test_node_flags_gt, raw_num_node_type, mmd_kernels)
        edge_type_mmd = eval_helper.compute_edge_type_mmd(final_samples_adjs, final_samples_adjs_gt, final_samples_node_flags, test_node_flags_gt, raw_num_adj_type, mmd_kernels)
        
        # node degree MMD
        logging.info(f'Node degree MMD: {node_deg_mmd}')

        # node type MMD
        logging.info(f'Node type MMD: {node_type_mmd}')

        # edge type MMD
        logging.info(f'Edge type MMD: {edge_type_mmd}')

        # triplet type metrics
        if not flag_node_only:
            logging.info("{} Evaluate triplet type TV using validation set statistics {}".format("="*10, "="*10))
            triplet_tv_dist_rej_val, triplet_tv_dist_all_val, triplet_tv_dist_full_val, triplet_novelty_val = eval_helper.compute_triplet_tv_dist(final_samples_adjs, final_samples_nodes, final_samples_node_flags, test_dl.test_triplet_dict, triplet_to_count)
            logging.info(f'Truncated TV distance rejecting novel triplets: {triplet_tv_dist_rej_val}')
            logging.info(f'Truncated TV distance accepting novel triplets: {triplet_tv_dist_all_val}')
            logging.info(f'Full TV distance considering novel and GT triplets: {triplet_tv_dist_full_val}')
            logging.info(f'Novel triplet percentage: {triplet_novelty_val}')

            logging.info("{} Evaluate triplet type TV using training set statistics {}".format("="*10, "="*10))
            triplet_tv_dist_rej_train, triplet_tv_dist_all_train, triplet_tv_dist_full_train, triplet_novelty_train = eval_helper.compute_triplet_tv_dist(final_samples_adjs, final_samples_nodes, final_samples_node_flags, test_dl.train_triplet_dict, triplet_to_count)
            logging.info(f'Truncated TV distance rejecting novel triplets: {triplet_tv_dist_rej_train}')
            logging.info(f'Truncated TV distance accepting novel triplets: {triplet_tv_dist_all_train}')
            logging.info(f'Full TV distance considering novel and GT triplets: {triplet_tv_dist_full_train}')
            logging.info(f'Novel triplet percentage: {triplet_novelty_train}')

        # bbox metrics
        if flag_bbox:
            # always use xyxy format for bounding box metrics computation
            pred_bbox = box_convert(final_samples_nodes_bbox, in_fmt='cxcywh', out_fmt='xyxy').clip(min=0.0, max=1.0)
            gt_bbox = box_convert(final_samples_nodes_bbox_gt, in_fmt='cxcywh', out_fmt='xyxy').clip(min=0.0, max=1.0)

            pred_bbox_blt_iou = eval_helper.compute_bbox_ioa(pred_bbox, final_samples_node_flags, canvas_size=32, flag_vanilla_iou=True, return_mean=True)
            gt_bbox_blt_iou = eval_helper.compute_bbox_ioa(gt_bbox, test_node_flags_gt,canvas_size=32, flag_vanilla_iou=True, return_mean=True)
            pred_bbox_blt_iou_percp = eval_helper.compute_bbox_ioa(pred_bbox, final_samples_node_flags, canvas_size=32, flag_perceptual_iou=True, return_mean=True)
            gt_bbox_blt_iou_percp = eval_helper.compute_bbox_ioa(gt_bbox, test_node_flags_gt, canvas_size=32, flag_perceptual_iou=True, return_mean=True)

            pred_bbox_blt_overlap = eval_helper.compute_bbox_ioa(pred_bbox, final_samples_node_flags, canvas_size=32, flag_overlap=True, return_mean=True)
            gt_bbox_blt_overlap = eval_helper.compute_bbox_ioa(gt_bbox, test_node_flags_gt, canvas_size=32, flag_overlap=True, return_mean=True)
            
            pred_bbox_blt_alignment = eval_helper.compute_bbox_ioa(pred_bbox, final_samples_node_flags, canvas_size=32, flag_alignment=True, return_mean=True)
            gt_bbox_blt_alignment = eval_helper.compute_bbox_ioa(gt_bbox, test_node_flags_gt, canvas_size=32, flag_alignment=True, return_mean=True)

            pred_bbox_self_metrics = {
                'iou_blt': pred_bbox_blt_iou,
                'iou_percp_blt': pred_bbox_blt_iou_percp,
                'overlap_blt': pred_bbox_blt_overlap,
                'alignment_blt': pred_bbox_blt_alignment
            }

            gt_bbox_self_metrics = {
                'iou_blt': gt_bbox_blt_iou,
                'iou_percp_blt': gt_bbox_blt_iou_percp,
                'overlap_blt': gt_bbox_blt_overlap,
                'alignment_blt': gt_bbox_blt_alignment
            }

            """bounding box metrics with matching"""
            weight_by_area = np.array([test_dl.bbox_area_stat[k] for k in sorted(test_dl.bbox_area_stat.keys())])  # area for node type 0, 1, 2, ...
            weight_by_area = weight_by_area / np.sum(weight_by_area)  # normalize to sum to 1

            weight_by_freq = np.array([test_dl.bbox_freq_stat[k] for k in sorted(test_dl.bbox_freq_stat.keys())])  # freq for node type 0, 1, 2, ...
            weight_by_freq = weight_by_freq / np.sum(weight_by_freq)  # normalize to sum to 1

            def _print_mat_f1_info(mat_f1, keyword=None):
                # mat_f1: [X, Y], X is number of generated samples, Y is number of gt samples
                matching_bbox_metrics = {
                    'avg_max_f1': mat_f1.max(axis=-1).mean(),
                    'avg_mean_f1': mat_f1.mean(axis=-1).mean(),
                    'avg_median_f1': np.median(mat_f1, axis=-1).mean(),
                }
                logging.info("{:s} F1 bbox metrics:".format(keyword if keyword is not None else ''))
                for k, v in matching_bbox_metrics.items():
                    logging.info("{}: {}".format(k, v))

            weights = [np.ones_like(weight_by_area), weight_by_area, weight_by_freq]
            mat_f1 = eval_helper.compute_bbox_f1(pred_bbox, final_samples_nodes, final_samples_node_flags, gt_bbox, final_samples_nodes_gt, test_node_flags_gt, weights)
            mat_f1_vanilla, mat_f1_area, mat_f1_freq = [np.squeeze(arr, axis=2) for arr in np.dsplit(mat_f1, 3)]

            dummy_x_gen = mask_nodes(torch.ones_like(final_samples_nodes), final_samples_node_flags)
            dummy_x_gt = mask_nodes(torch.ones_like(final_samples_nodes_gt), test_node_flags_gt)
            mat_f1_no_node_type = eval_helper.compute_bbox_f1(pred_bbox, dummy_x_gen, final_samples_node_flags, gt_bbox, dummy_x_gt, test_node_flags_gt, class_weight_ls=None).squeeze(2)
            
            matching_bbox_metrics = {
                'vanilla_f1_avg_max': mat_f1_vanilla.max(axis=-1).mean(),
                'vanilla_f1_avg_mean': mat_f1_vanilla.mean(axis=-1).mean(),
                'vanilla_f1_avg_median': np.median(mat_f1_vanilla, axis=-1).mean(),
                'area_f1_avg_max': mat_f1_area.max(axis=-1).mean(),
                'area_f1_avg_mean': mat_f1_area.mean(axis=-1).mean(),
                'area_f1_avg_median': np.median(mat_f1_area, axis=-1).mean(),
                'freq_f1_avg_max': mat_f1_freq.max(axis=-1).mean(),
                'freq_f1_avg_mean': mat_f1_freq.mean(axis=-1).mean(),
                'freq_f1_avg_median': np.median(mat_f1_freq, axis=-1).mean(),
                'no_node_type_f1_avg_max': mat_f1_no_node_type.max(axis=-1).mean(),
                'no_node_type_f1_avg_mean': mat_f1_no_node_type.mean(axis=-1).mean(),
                'no_node_type_f1_avg_median': np.median(mat_f1_no_node_type, axis=-1).mean(),
            }

            # plot scene graphs with the closest retrieved graphs
            _print_mat_f1_info(mat_f1_vanilla, keyword='Vanilla')
            _print_mat_f1_info(mat_f1_area, keyword='Area weighted')
            _print_mat_f1_info(mat_f1_freq, keyword='Freq weighted')
            _print_mat_f1_info(mat_f1_no_node_type, keyword='No node type')

            logging.info("Making plots for scene graphs with bbox...")
            plot_scene_graph_bbox(final_samples_nodes, final_samples_nodes_bbox, final_samples_adjs,
                                    final_samples_nodes_gt, final_samples_nodes_bbox_gt, final_samples_adjs_gt,
                                    mat_f1_vanilla, final_samples_node_flags, test_node_flags_gt, idx_to_word, 
                                    save_dir=sg_plot_path, title='bbox_vanilla_f1_' + fig_title, num_plots=8)
            
            plot_scene_graph_bbox(final_samples_nodes, final_samples_nodes_bbox, final_samples_adjs,
                                    final_samples_nodes_gt, final_samples_nodes_bbox_gt, final_samples_adjs_gt,
                                    mat_f1_area, final_samples_node_flags, test_node_flags_gt, idx_to_word, 
                                    save_dir=sg_plot_path, title='bbox_area_f1_' + fig_title, num_plots=8)
            
            plot_scene_graph_bbox(final_samples_nodes, final_samples_nodes_bbox, final_samples_adjs,
                                    final_samples_nodes_gt, final_samples_nodes_bbox_gt, final_samples_adjs_gt,
                                    mat_f1_freq, final_samples_node_flags, test_node_flags_gt, idx_to_word, 
                                    save_dir=sg_plot_path, title='bbox_freq_f1_' + fig_title, num_plots=8)
            
            plot_scene_graph_bbox(final_samples_nodes, final_samples_nodes_bbox, final_samples_adjs,
                                    final_samples_nodes_gt, final_samples_nodes_bbox_gt, final_samples_adjs_gt,
                                    mat_f1_no_node_type, final_samples_node_flags, test_node_flags_gt, idx_to_word, 
                                    save_dir=sg_plot_path, title='bbox_no_node_type_f1_' + fig_title, num_plots=8)
            

        # save to tensorboard
        if writer is not None:
            for kernel, val_par in node_deg_mmd.items():
                for key, val in val_par.items():
                    writer.add_scalar(f'gen_epoch/node_{key}_mmd_{kernel}', val, epoch)
            for kernel, val in node_type_mmd.items():
                writer.add_scalar(f'gen_epoch/node_type_mmd_{kernel}', val, epoch)
            for kernel, val in edge_type_mmd.items():
                writer.add_scalar(f'gen_epoch/edge_type_mmd_{kernel}', val, epoch)
            if not flag_node_only:
                writer.add_scalar('gen_epoch/triplet_tv_dist_rej_val', triplet_tv_dist_rej_val, epoch)
                writer.add_scalar('gen_epoch/triplet_tv_dist_all_val', triplet_tv_dist_all_val, epoch)
                writer.add_scalar('gen_epoch/triplet_tv_dist_full_val', triplet_tv_dist_full_val, epoch)
                writer.add_scalar('gen_epoch/triplet_novelty_val', triplet_novelty_val, epoch)

                writer.add_scalar('gen_epoch/triplet_tv_dist_rej_train', triplet_tv_dist_rej_train, epoch)
                writer.add_scalar('gen_epoch/triplet_tv_dist_all_train', triplet_tv_dist_all_train, epoch)
                writer.add_scalar('gen_epoch/triplet_tv_dist_full_train', triplet_tv_dist_full_train, epoch)
                writer.add_scalar('gen_epoch/triplet_novelty_train', triplet_novelty_train, epoch)

            if flag_bbox:
                for key in pred_bbox_self_metrics.keys():
                    writer.add_scalar(f'gen_epoch/pred_bbox_self_{key}', pred_bbox_self_metrics[key], epoch)
                    writer.add_scalar(f'gen_epoch/gt_bbox_self_{key}', gt_bbox_self_metrics[key], epoch)
                for key in matching_bbox_metrics.keys():
                    writer.add_scalar(f'gen_epoch/{key}', matching_bbox_metrics[key], epoch)

        # save the final samples including evaluation results
        np.savez_compressed(path_final_samples_array,
                            samples_node_flags=final_samples_node_flags.cpu().bool().numpy(),
                            samples_a=final_samples_adjs.cpu().numpy(),
                            samples_x=final_samples_nodes.cpu().numpy(),
                            raw_a=final_raw_adjs.cpu().numpy(),
                            raw_x=final_raw_nodes.cpu().numpy(),
                            gt_node_flags=test_node_flags_gt.cpu().bool().numpy(),
                            gt_a=final_samples_adjs_gt.cpu().numpy(),
                            gt_x=final_samples_nodes_gt.cpu().numpy(),
                            samples_x_bbox=final_samples_nodes_bbox.cpu().numpy() if flag_bbox else None,
                            gt_x_bbox=final_samples_nodes_bbox_gt.cpu().numpy() if flag_bbox else None,
                            mat_f1_vanilla=mat_f1_vanilla if flag_bbox else None,
                            mat_f1_area=mat_f1_area if flag_bbox else None,
                            mat_f1_freq=mat_f1_freq if flag_bbox else None,
                            mat_f1_no_node_type=mat_f1_no_node_type if flag_bbox else None,
                            )

        # save to csv
        if sampling_params is not None:
            result_dict = {
                'gen_data_size': len(final_samples_adjs),
                'test_data_size': len(final_samples_adjs_gt),
            }
            for kernel, val_par in node_deg_mmd.items():
                for key, val in val_par.items():
                    result_dict[f'node_{key}_mmd_{kernel}'] = val
            for kernel, val in node_type_mmd.items():
                result_dict[f'node_type_mmd_{kernel}'] = val
            for kernel, val in edge_type_mmd.items():
                result_dict[f'edge_type_mmd_{kernel}'] = val
            if not flag_node_only:
                result_dict.update({
                    'triplet_tv_dist_rej_val': triplet_tv_dist_rej_val,
                    'triplet_tv_dist_all_val': triplet_tv_dist_all_val,
                    'triplet_tv_dist_full_val': triplet_tv_dist_full_val,
                    'triplet_novelty_val': triplet_novelty_val,
                    'triplet_tv_dist_rej_train': triplet_tv_dist_rej_train,
                    'triplet_tv_dist_all_train': triplet_tv_dist_all_train,
                    'triplet_tv_dist_full_train': triplet_tv_dist_full_train,
                    'triplet_novelty_train': triplet_novelty_train,
                })
            sampling_params.update(result_dict)

            if flag_bbox:
                bbox_dict = {
                    'pred_iou_blt': pred_bbox_self_metrics['iou_blt'],
                    'pred_iou_percp_blt': pred_bbox_self_metrics['iou_percp_blt'],
                    'pred_overlap_blt': pred_bbox_self_metrics['overlap_blt'],
                    'pred_alignment_blt': pred_bbox_self_metrics['alignment_blt'],
                    'gt_iou_blt': gt_bbox_self_metrics['iou_blt'],
                    'gt_iou_percp_blt': gt_bbox_self_metrics['iou_percp_blt'],
                    'gt_overlap_blt': gt_bbox_self_metrics['overlap_blt'],
                    'gt_alignment_blt': gt_bbox_self_metrics['alignment_blt'],
                    'vanilla_f1_avg_max': matching_bbox_metrics['vanilla_f1_avg_max'],
                    'vanilla_f1_avg_mean': matching_bbox_metrics['vanilla_f1_avg_mean'],
                    'vanilla_f1_avg_median': matching_bbox_metrics['vanilla_f1_avg_median'],
                    'area_f1_avg_max': matching_bbox_metrics['area_f1_avg_max'],
                    'area_f1_avg_mean': matching_bbox_metrics['area_f1_avg_mean'],
                    'area_f1_avg_median': matching_bbox_metrics['area_f1_avg_median'],
                    'freq_f1_avg_max': matching_bbox_metrics['freq_f1_avg_max'],
                    'freq_f1_avg_mean': matching_bbox_metrics['freq_f1_avg_mean'],
                    'freq_f1_avg_median': matching_bbox_metrics['freq_f1_avg_median'],
                    'no_node_type_f1_avg_max': matching_bbox_metrics['no_node_type_f1_avg_max'],
                    'no_node_type_f1_avg_mean': matching_bbox_metrics['no_node_type_f1_avg_mean'],
                    'no_node_type_f1_avg_median': matching_bbox_metrics['no_node_type_f1_avg_median'],
                }
                sampling_params.update(bbox_dict)

            df = pd.DataFrame.from_dict(data=sampling_params, orient='index').transpose()
            mmd_keys = sorted([item for item in df.columns.tolist() if 'mmd' in item])
            triplet_keys = sorted([item for item in df.columns.tolist() if 'triplet' in item])
            cols = ['model_nm', 'weight_kw', 'gen_data_size', 'test_data_size',
                    *mmd_keys, *triplet_keys,
                    'model_path']
            if flag_node_only:
                cols_to_rm = [*triplet_keys, 'edge_type_mmd']
                cols = [col for col in cols if col not in cols_to_rm]
            if flag_bbox:
                allowed_keys = ['no_node_type_f1_avg_max', 'vanilla_f1_avg_max', 'area_f1_avg_max', 'freq_f1_avg_max',
                                'pred_iou_percp_blt', 'pred_iou_blt', 'pred_overlap_blt', 'pred_alignment_blt',
                                'gt_iou_percp_blt', 'gt_iou_blt', 'gt_overlap_blt', 'gt_alignment_blt',
                                'no_node_type_f1_avg_mean', 'vanilla_f1_avg_mean', 'area_f1_avg_mean', 'freq_f1_avg_mean',
                                'no_node_type_f1_avg_min', 'vanilla_f1_avg_min', 'area_f1_avg_min', 'freq_f1_avg_min',
                                ]
                bbox_start_idx = 4
                for key in allowed_keys:
                    if key in sampling_params:
                        cols.insert(bbox_start_idx, key)
                        bbox_start_idx += 1

            df = df[cols]
            csv_path = os.path.join(config.logdir, 'eval_results.csv')
            df.to_csv(csv_path, header=not os.path.exists(csv_path), index=False, mode='a')

        # print out scene graphs in strings
        sg_str_ls = []
        for i_graph, (sample_a, sample_x) in enumerate(zip(final_samples_adjs, final_samples_nodes)):
            sample_a, sample_x = sample_a.long(), sample_x.long()
            num_nodes = final_samples_node_flags[i_graph].gt(0).sum()  # int
            sg_str = '{:s} scene graph no. {:d} / {:d} {:s}'.format('-' * 40, i_graph, len(final_samples_adjs),
                                                                    '-' * 40) + '\n'
            sg_str += "".ljust(20)
            for idx_j in range(num_nodes):
                sg_str += idx_to_word['ind_to_classes'][sample_x[idx_j]].ljust(20)
            sg_str += '\n'

            for idx_i in range(num_nodes):
                sg_str += idx_to_word['ind_to_classes'][sample_x[idx_i]].ljust(20)
                for idx_j in range(num_nodes):
                    if sample_a[idx_i][idx_j] > 0:
                        sg_str += idx_to_word['ind_to_predicates'][sample_a[idx_i][idx_j]].ljust(20)
                    else:
                        sg_str += "".ljust(20)
                sg_str += '\n'
            sg_str_ls.append(sg_str)

        np.savetxt(save_path_sg_txt, sg_str_ls, fmt='%s')

    # clean up
    del test_adjs_gt, test_node_flags, sampler_dl
