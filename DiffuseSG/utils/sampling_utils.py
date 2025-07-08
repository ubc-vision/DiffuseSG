import logging
import torch

from utils.visual_utils import plot_graphs_adj
from runner.mcmc_sampler.edm import NodeAdjEDMSampler


def get_mc_sampler(config):
    """
    Configure MCMC sampler.
    """
    # Setup sampler
    flag_clip_samples = config.mcmc.sample_clip.min is not None and config.mcmc.sample_clip.max is not None
    assert config.mcmc.name == 'edm'
    mc_sampler = NodeAdjEDMSampler(num_steps=config.mcmc.num_steps,
                                    clip_samples=flag_clip_samples,
                                    clip_samples_min=config.mcmc.sample_clip.min,
                                    clip_samples_max=config.mcmc.sample_clip.max,
                                    clip_samples_scope=config.mcmc.sample_clip.scope,
                                    dev=config.dev,
                                    objective='edm',
                                    self_condition=config.train.self_cond,
                                    symmetric_noise=False)

    # Print out sampler information
    logging.info('EDM-variant objective. \n'
                    'Model: {:s}. Num of steps: {:d}'.format(config.mcmc.name, config.mcmc.num_steps))

    logging.info('Self-conditioning: {}'.format(config.train.self_cond))

    return mc_sampler


def load_model(ckp_data, model, weight_keyword):
    """
    Load network weight.
    """
    assert weight_keyword in ckp_data
    cur_keys = sorted(list(model.state_dict().keys()))
    ckp_keys = sorted(list(ckp_data[weight_keyword].keys()))
    if set(cur_keys) == set(cur_keys) & set(ckp_keys):
        model.load_state_dict(ckp_data[weight_keyword], strict=True)
    else:
        to_load_state_dict = {}
        for cur_key, ckp_key in zip(cur_keys, ckp_keys):
            if cur_key == ckp_key:
                pass
            # note: .module prefix is added during the DP training
            elif cur_key.startswith('module.') and not ckp_key.startswith('module.'):
                assert cur_key == 'module.' + ckp_key
            elif ckp_key.startswith('module.') and not cur_key.startswith('module.'):
                assert 'module.' + cur_key == ckp_key
            else:
                raise NotImplementedError
            to_load_state_dict[cur_key] = ckp_data[weight_keyword][ckp_key]
        assert set(cur_keys) == set(list(to_load_state_dict.keys()))
        model.load_state_dict(to_load_state_dict, strict=True)
        del to_load_state_dict
        torch.cuda.empty_cache()
    return model


def eval_sample_batch(sample_b, test_adj_b, init_adjs, save_dir, title="", threshold=0.5):
    """
    Evaluate the graph data in torch tensor.
    """
    delta = sample_b - test_adj_b
    init_delta = init_adjs - test_adj_b
    round_init_adjs = torch.where(init_adjs < threshold, torch.zeros_like(init_adjs), torch.ones_like(init_adjs))
    round_init_delta = round_init_adjs - test_adj_b
    logging.info(f"sample delta_norm_mean: {delta.norm(dim=[1, 2]).mean().item():.3e} "
                 f"| init delta_norm_mean: {init_delta.norm(dim=[1, 2]).mean().item():.3e}"
                 f"| round init delta_norm_mean: {round_init_delta.norm(dim=[1, 2]).mean().item():.3e}")

    plot_graphs_adj(sample_b,
                    node_num=test_adj_b.sum(-1).gt(1e-5).sum(-1).cpu().numpy(),
                    title=title,
                    save_dir=save_dir)
