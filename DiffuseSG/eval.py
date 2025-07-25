import os
import copy
import torch
import logging


from runner.sampler.sampler_node_adj import sg_go_sampling
from utils.arg_parser import set_seed_and_logger, parse_arguments, backup_code
from utils.learning_utils import get_network
from utils.dataloader import load_data
from utils.sampling_utils import get_mc_sampler, load_model
from utils.dist_training import DistributedHelper


def get_ema_weight_keywords(ckp_data, args_use_ema):
    all_weight_keywords = []
    for item in list(ckp_data.keys()):
        if item.startswith('model'):
            all_weight_keywords.append(item)

    weight_keywords = ['model']
    if args_use_ema is None:
        logging.info('Not using EMA weight.')
    elif args_use_ema == 'all':
        # lazy init: to use all online and EMA weights
        weight_keywords = all_weight_keywords
        logging.info('Use all possible EMA weights.')
    else:
        logging.info('Using EMA weight with coefficients: {}'.format(args_use_ema))
        if 1.0 not in args_use_ema:
            weight_keywords.remove('model')
        else:
            args_use_ema.remove(1.0)
        for item in args_use_ema:
            _weight_keyword = 'model_ema_beta_{:.4f}'.format(item)
            assert _weight_keyword in all_weight_keywords, "{} not found in the model data!".format(_weight_keyword)
            weight_keywords.append(_weight_keyword)

    logging.info('Model weights to load: {}'.format(weight_keywords))
    return weight_keywords


def batch_evaluate(model, dist_helper, test_dl, mc_sampler, config, args_model_path, args_use_ema, writer):
    logging.info("Models to load:")
    [logging.info("{:d}: {:s}".format(i, item)) for i, item in enumerate(args_model_path)]

    triplet_to_count = list(test_dl.test_triplet_dict.keys())  # use all triplets for TV computation

    for model_path in args_model_path:
        model_nm = os.path.basename(model_path)
        logging.info("{:s} Evaluating model at {:s} {:s}".format('-' * 6, model_path, '-' * 6))
        ckp_data = torch.load(model_path, map_location=lambda storage, loc: storage)
        weight_keywords = get_ema_weight_keywords(ckp_data, copy.copy(args_use_ema))

        for weight_kw in weight_keywords:
            logging.info("Loading weight for {:s} to create samples...".format(weight_kw))
            load_model(ckp_data, model, weight_kw)

            sampling_params = {'model_nm': model_nm, 'weight_kw': weight_kw, 'model_path': model_path}
            epoch = int(model_nm.split('_')[-1].replace('.pth', ''))

            # Go sampling!
            flag_node_only = config.train.node_only
            flag_binary_edge = config.train.binary_edge
            pkl_data = test_dl.pkl_data
            idx_to_word = test_dl.idx_to_word
            skip_eval = config.test.skip_eval
            random_node_num = config.test.random_node_num
            sg_go_sampling(epoch=epoch, model=model, dist_helper=dist_helper, eval_mode=True,
                            test_dl=test_dl, mc_sampler=mc_sampler, config=config, sanity_check=False,
                            sampling_params=sampling_params,
                            triplet_to_count=triplet_to_count, flag_node_only=flag_node_only, flag_binary_edge=flag_binary_edge,
                            pkl_data=pkl_data, idx_to_word=idx_to_word, writer=writer, skip_eval=skip_eval, random_node_num=random_node_num)

        # sync DDP processes and release GPU memory
        dist_helper.ddp_sync()
        del ckp_data


def evaluate_main():
    args, config = parse_arguments(mode='eval')
    dist_helper = DistributedHelper(args.dp, args.ddp, args.ddp_gpu_ids, args.ddp_init_method)

    writer = set_seed_and_logger(config, args.log_level, args.comment, dist_helper, eval_mode=True)
    backup_code(config, args.config_file)
    logging.info(args)

    # Load dataset
    train_dl, test_dl = load_data(config, dist_helper, eval_mode=True)

    # Initialize MCMC sampler
    mc_sampler = get_mc_sampler(config)

    # Initialize network
    model = get_network(config, dist_helper)

    # Load model
    batch_evaluate(model, dist_helper, test_dl, mc_sampler, config, args.model_path, args.use_ema, writer=writer)

    # Post-training sampling
    logging.info('EVALUATION IS FINISHED.')


if __name__ == "__main__":
    evaluate_main()
