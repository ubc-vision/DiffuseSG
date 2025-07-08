import logging

from runner.trainer.trainer_node_adj import node_adj_go_training
from utils.arg_parser import backup_code, parse_arguments, set_seed_and_logger
from utils.dataloader import load_data
from utils.dist_training import DistributedHelper
from utils.learning_utils import (get_ema_helper, get_network, get_optimizer,
                                  get_rainbow_loss,
                                  get_training_objective_generator)
from utils.sampling_utils import get_mc_sampler


def init_basics(mode="train"):
    """Initialize basic components for training or evaluation.

    This function sets up the basic components needed for training or evaluation,
    including argument parsing, distributed training setup, logging, and code backup.

    Args:
        mode (str): Either 'train' or 'eval' to specify the operation mode.

    Returns:
        tuple: A tuple containing:
            - args: Parsed command line arguments
            - config: Configuration object
            - dist_helper: Distributed training helper
            - writer: TensorBoard writer for logging
    """
    # Initialization
    args, config = parse_arguments(mode=mode)
    dist_helper = DistributedHelper(args.dp, args.ddp, args.ddp_gpu_ids, args.ddp_init_method)
    writer = set_seed_and_logger(
        config, args.log_level, args.comment, dist_helper, eval_mode=mode == "eval"
    )
    backup_code(config, args.config_file)
    return args, config, dist_helper, writer


def init_model(config, dist_helper):
    """Initialize the model and training components.

    This function sets up all the components needed for training, including
    the model, optimizer, training objective generator, MCMC sampler, and loss function.

    Args:
        config: Configuration object containing model and training parameters
        dist_helper: Distributed training helper for handling multi-GPU training

    Returns:
        tuple: A tuple containing:
            - train_obj_gen: Training objective generator
            - mc_sampler: MCMC sampler for the diffusion process
            - model: The neural network model
            - optimizer: Optimizer for training
            - scheduler: Learning rate scheduler
            - ema_helper: EMA model averaging helper
            - loss_func: Loss function for training
    """
    # Initialize training objective generator
    train_obj_gen = get_training_objective_generator(config)

    # Initialize MCMC sampler
    mc_sampler = get_mc_sampler(config)

    # Initialize network model & optimizer
    model = get_network(config, dist_helper)
    optimizer, scheduler = get_optimizer(model, config, dist_helper)

    # Initialize EMA helper
    ema_helper = get_ema_helper(config, model)

    # Initialize loss function
    loss_func = get_rainbow_loss(config)
    return train_obj_gen, mc_sampler, model, optimizer, scheduler, ema_helper, loss_func


def main():
    """Main training function.

    This function orchestrates the entire training process:
    1. Initializes basic components (args, config, distributed training, logging)
    2. Loads the dataset
    3. Initializes the model and training components
    4. Starts the training process
    5. Cleans up distributed training resources

    The training process uses distributed data parallel (DDP) training when specified,
    and includes features like EMA model averaging and periodic evaluation.
    """
    """Initialize basics"""
    args, config, dist_helper, writer = init_basics()

    """Get dataloader"""
    train_dl, test_dl = load_data(config, dist_helper)

    """Get network"""
    train_obj_gen, mc_sampler, model, optimizer, scheduler, ema_helper, loss_func = init_model(
        config, dist_helper
    )

    """Go training"""
    node_adj_go_training(
        model,
        optimizer,
        scheduler,
        ema_helper,
        train_dl,
        test_dl,
        train_obj_gen,
        loss_func,
        mc_sampler,
        config,
        dist_helper,
        writer,
    )

    # Clean up DDP utilities after training
    dist_helper.clean_up()

    logging.info("TRAINING IS FINISHED.")


if __name__ == "__main__":
    main()
