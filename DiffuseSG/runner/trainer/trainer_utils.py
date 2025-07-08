import copy
import logging
import os
import time

import numpy as np

from utils.dist_training import get_ddp_save_flag, dist_save_model


def get_logger_per_epoch(epoch, flag_node_adj):
    """
    Create dict to save learning status at the beginning of each epoch.
    """
    _loss_status = {
        'summed_loss': [],
        'time_start': None,
        'time_elapsed': None,
        'noise_label': []
    }
    if flag_node_adj:
        _loss_status['reg_loss_adj'] = []
        _loss_status['reg_loss_node'] = []
    else:
        _loss_status['regression_loss'] = []

    loss_status_ls = [copy.deepcopy(_loss_status) for _ in range(2)]

    logger = {'train': loss_status_ls[0],
              'test': loss_status_ls[1],
              'epoch': epoch,
              'lr': 0.0}
    return logger


def update_epoch_learning_status(epoch_logger, mode, reg_loss=None,
                                 reg_loss_adj=None, reg_loss_node=None, noise_label=None):
    """
    Update learning status dict.
    """
    assert mode == 'train' or 'test'

    if reg_loss is not None:
        assert reg_loss_adj is None and reg_loss_node is None
        epoch_logger[mode]['regression_loss'].append(reg_loss.cpu().numpy())
        epoch_logger[mode]['summed_loss'].append(reg_loss.cpu().numpy())
    else:
        epoch_logger[mode]['reg_loss_adj'].append(reg_loss_adj.cpu().numpy())
        epoch_logger[mode]['reg_loss_node'].append(reg_loss_node.cpu().numpy())
        epoch_logger[mode]['summed_loss'].append((reg_loss_adj + reg_loss_node).cpu().numpy())

    epoch_logger[mode]['noise_label'].append(noise_label.cpu().numpy())
    if epoch_logger[mode]['time_start'] is None:
        epoch_logger[mode]['time_start'] = time.time()
    else:
        # update each time for convenience, only the last timestamp is useful
        epoch_logger[mode]['time_elapsed'] = time.time() - epoch_logger[mode]['time_start']
    return epoch_logger


def print_epoch_learning_status(epoch_logger, f_train_loss, f_test_loss, writer, objective, flag_node_adj):
    """
    Show the learning status of this epoch.
    """
    epoch = epoch_logger['epoch']
    lr = epoch_logger['lr']

    def _write_to_file_handler(np_array_data, file_handler, line_sampling_freq):
        for i_line, line in enumerate(np_array_data):
            if i_line % line_sampling_freq == 0:
                line_str = np.array2string(line, formatter={'float_kind': lambda x: "%.6f" % x}, separator=" ")
                file_handler.write(line_str[1:-1] + '\n')
        file_handler.flush()

    for mode, f_handler in zip(['train', 'test'], [f_train_loss, f_test_loss]):

        flag_empty = len(epoch_logger[mode]['summed_loss']) == 0

        if not flag_empty:
            summed_loss = np.concatenate(epoch_logger[mode]['summed_loss'])  # array, [N]
            time_elapsed = epoch_logger[mode]['time_elapsed']  # scalar
            noise_label = np.concatenate(epoch_logger[mode]['noise_label'])  # array, [N]
            i_iter = epoch_logger['epoch'] * len(summed_loss)

            if flag_node_adj:
                reg_loss_node = np.concatenate(epoch_logger[mode]['reg_loss_node'])  # array, [N]
                reg_loss_adj = np.concatenate(epoch_logger[mode]['reg_loss_adj'])
                logging.info(f'epoch: {epoch:05d}| {mode:5s} | '
                             f'total loss: {np.mean(summed_loss):10.6f} | '
                             f'{objective:s} adj_loss: {np.mean(reg_loss_adj):10.6f} | '
                             f'node_loss: {np.mean(reg_loss_node):10.6f} | '
                             f'time: {time_elapsed:5.2f}s | ')

                down_sampling_freq = 1000
                if get_ddp_save_flag():
                    # record epoch-wise and sample-wise training status into tensorboard
                    cat_loss = np.stack([noise_label, reg_loss_adj, reg_loss_node], axis=1)  # array, [N, X]
                    writer.add_scalar("{:s}_epoch/loss_adj".format(mode), np.mean(reg_loss_adj), epoch)
                    writer.add_scalar("{:s}_epoch/loss_node".format(mode), np.mean(reg_loss_node), epoch)
                    if mode == 'train':
                        writer.add_scalar("{:s}_epoch/learning_rate".format(mode), lr, epoch)
                    for i in range(len(cat_loss)):
                        if i % down_sampling_freq == 0:
                            writer.add_scalar("{:s}_sample/loss_adj".format(mode), reg_loss_adj[i], i + i_iter)
                            writer.add_scalar("{:s}_sample/loss_node".format(mode), reg_loss_node[i], i + i_iter)
                            writer.add_scalar("{:s}_sample/noise_label".format(mode), noise_label[i], i + i_iter)
                    writer.flush()
            else:
                regression_loss = np.concatenate(epoch_logger[mode]['regression_loss'])  # array, [N]
                logging.info(f'epoch: {epoch:05d}| {mode:5s} | '
                             f'total loss: {np.mean(summed_loss):10.6f} | '
                             f'{objective:s} loss: {np.mean(regression_loss):10.6f} | '
                             f'time: {time_elapsed:5.2f}s | ')

                down_sampling_freq = 1
                if get_ddp_save_flag():
                    # record epoch-wise and sample-wise training status into tensorboard
                    cat_loss = np.stack([noise_label, regression_loss], axis=1)  # array, [N, X]
                    writer.add_scalar("{:s}_epoch/loss".format(mode), np.mean(regression_loss), epoch)
                    if mode == 'train':
                        writer.add_scalar("{:s}_epoch/learning_rate".format(mode), lr, epoch)
                    for i in range(len(cat_loss)):
                        writer.add_scalar("{:s}_sample/loss".format(mode), regression_loss[i], i + i_iter)
                        writer.add_scalar("{:s}_sample/noise_label".format(mode), noise_label[i], i + i_iter)
                    writer.flush()

            if get_ddp_save_flag():
                # record sample-wise training status into txt file
                _write_to_file_handler(cat_loss, f_handler, down_sampling_freq)


def check_best_model(model, ema_helper, epoch_logger, best_model_status, save_interval, config, dist_helper):
    """
    Check if the latest training leads to a better model.
    """
    if get_ddp_save_flag():
        lowest_loss = best_model_status["loss"]
        mean_train_loss = np.concatenate(epoch_logger['train']['summed_loss']).mean()
        mean_test_loss = np.concatenate(epoch_logger['test']['summed_loss']).mean()
        epoch = epoch_logger['epoch']
        if lowest_loss > mean_test_loss and epoch > save_interval:
            best_model_status["epoch"] = epoch
            best_model_status["loss"] = mean_test_loss
            to_save = get_ckpt_data(model, ema_helper, epoch, mean_train_loss, mean_test_loss, config, dist_helper)

            # save to model checkpoint dir (many network weights)
            to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_best.pth")
            dist_save_model(to_save, to_save_path)
            logging.info(f"epoch: {epoch:05d}| best model updated at {to_save_path:s}")

            # save to best model storage directory (single network weight)
            to_save_path = os.path.join(config.model_save_dir, f"{config.dataset.name}_best.pth")
            dist_save_model(to_save, to_save_path)


def save_ckpt_model(model, ema_helper, epoch_logger, config, dist_helper):
    """
    Save the checkpoint weight.
    """
    mean_train_loss = np.concatenate(epoch_logger['train']['summed_loss']).mean()
    mean_test_loss = np.concatenate(epoch_logger['test']['summed_loss']).mean()
    epoch = epoch_logger['epoch']
    to_save = get_ckpt_data(model, ema_helper, epoch, mean_train_loss, mean_test_loss, config, dist_helper)
    to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_{epoch:05d}.pth")
    dist_save_model(to_save, to_save_path)


def get_ckpt_data(model, ema_helper, epoch, train_loss, test_loss, config, dist_helper):
    """
    Create a dictionary containing necessary stuff to save.
    """
    to_save = {
        'model': model.state_dict(),
        'config': config.to_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

    if ema_helper is not None:
        for ema in ema_helper:
            beta = ema.beta
            to_save['model_ema_beta_{:.4f}'.format(beta)] = ema.ema_model.state_dict()

    return to_save

