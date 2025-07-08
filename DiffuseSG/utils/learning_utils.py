import logging
import os

import torch
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer

from ema_pytorch import EMA
from model.precond.precond import Precond, NodeAdjPrecond
from model.diffusesg.diffusesg import DiffuseSG
from model.self_cond.self_cond_wrapper import SelfCondWrapper

from loss.rainbow_loss import NodeAdjRainbowLoss
from runner.objectives.edm import NodeAdjEDMObjectiveGenerator
from utils.dist_training import get_ddp_save_flag
from utils.sampling_utils import load_model
from utils.sg_utils import get_node_adj_model_input_output_channels


def get_training_objective_generator(config):
    """
    Get training objective generator.
    """
    assert config.mcmc.name == "edm"
    train_obj_gen = NodeAdjEDMObjectiveGenerator(precond=config.mcmc.precond,
                                                    sigma_dist=config.mcmc.sigma_dist,
                                                    other_params=config.mcmc,
                                                    dev=config.dev,
                                                    symmetric_noise=False)
    return train_obj_gen


def get_network(config, dist_helper):
    """
    Configure the neural network.
    """
    model_config = config.model
    feature_nums = model_config.feature_dims if 'feature_dims' in model_config else [0]

    plot_save_dir = os.path.join(config.logdir, 'training_plot')
    if get_ddp_save_flag():
        os.makedirs(plot_save_dir, exist_ok=True)
    if config.model.name in ['diffuse_sg']:
        # with node and edge attributes
        in_chans, out_chans_adj, out_chans_node = get_node_adj_model_input_output_channels(config)

        denoising_model = DiffuseSG(
            img_size=config.dataset.max_node_num,
            in_chans=in_chans,
            # patch_size=4,
            # embed_dim=96,
            # depths=[2, 2, 6, 2],
            patch_size=model_config.patch_size,
            embed_dim=feature_nums[-1],
            depths=model_config.depths,
            num_heads=[3, 6, 12, 24],
            window_size=model_config.window_size,
            mlp_ratio=4.,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
            self_condition=config.train.self_cond,
            symmetric_noise=not config.flag_sg,
            out_chans_adj=out_chans_adj,
            out_chans_node=out_chans_node
        ).to(config.dev)
    else:
        raise ValueError(f'Unknown model name {config.model.name}')

    # EDM preconditioning module adaptation
    if config.mcmc.name == 'edm':
        if config.flag_sg:
            denoising_model = NodeAdjPrecond(precond=config.mcmc.precond,
                                             model=denoising_model,
                                             self_condition=config.train.self_cond,
                                             symmetric_noise=not config.flag_sg)
        else:
            denoising_model = Precond(precond=config.mcmc.precond,
                                      model=denoising_model,
                                      self_condition=config.train.self_cond)

    # non-EDM self-conditioning nn.Module wrapper
    # EDM doesn't need this as its precond layer is already an nn.Module layer
    if config.mcmc.name != 'edm' and config.train.self_cond:
        denoising_model = SelfCondWrapper(model=denoising_model, self_condition=config.train.self_cond)

    # DEBUG: plot model intermediate states
    denoising_model.plot_save_dir = plot_save_dir

    # count model parameters
    logging.info('model: ' + str(denoising_model))
    param_string, total_params, total_trainable_params = count_model_params(denoising_model)
    logging.info(f"Parameters: \n{param_string}")
    logging.info(f"Parameters Count: {total_params:,}, Trainable: {total_trainable_params:,}")

    # load checkpoint to resume training
    if config.train.resume is not None:
        logging.info("Resuming training from checkpoint: {:s}".format(config.train.resume))
        ckp_data = torch.load(config.train.resume)
        denoising_model = load_model(ckp_data, denoising_model, 'model')

    # adapt to distributed training
    if dist_helper.is_distributed:
        denoising_model = dist_helper.dist_adapt_model(denoising_model)
    else:
        logging.info("Distributed OFF. Single-GPU training.")

    return denoising_model


def count_model_params(model):
    """
    Go through the model parameters
    """
    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_string, total_params, total_trainable_params


def get_optimizer(model, config, dist_helper):
    """
    Configure the optimizer.
    """
    if dist_helper.is_ddp:
        optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                            optimizer_class=torch.optim.Adam,
                                            lr=config.train.lr_init,
                                            betas=(0.9, 0.999), eps=1e-8,
                                            weight_decay=config.train.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.train.lr_init,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    return optimizer, scheduler


def get_ema_helper(config, model):
    """
    Setup exponential moving average training helper.
    """
    flag_ema = False
    ema_coef = config.train.ema_coef
    if isinstance(ema_coef, list):
        flag_ema = True
    if isinstance(ema_coef, float):
        flag_ema = config.train.ema_coef < 1
    if flag_ema:
        ema_coef = [ema_coef] if isinstance(ema_coef, float) else ema_coef
        assert isinstance(ema_coef, list)
        ema_helper = []
        for coef in sorted(ema_coef):
            ema = EMA(model=model, beta=coef, update_every=1, update_after_step=0, inv_gamma=1, power=1)
            ema_helper.append(ema)
        logging.info("Exponential moving average is ON. Coefficient: {}".format(ema_coef))
    else:
        ema_helper = None
        logging.info("Exponential moving average is OFF.")
    return ema_helper


def get_rainbow_loss(config):
    """
    Construct all-in-one training loss wrapper.
    """

    assert config.flag_sg
    loss_func = NodeAdjRainbowLoss(edge_loss_weight=config.train.edge_loss_weight,
                                    node_loss_weight=config.train.node_loss_weight,
                                    flag_reweight=config.train.reweight_entry,
                                    objective=config.mcmc.name)

    logging.info("Loss weight: denoising regression loss: {:.2f}".format(1.0))

    logging.info("Loss reweight based on zero/one entries: {}.".format(config.train.reweight_entry))
    return loss_func
