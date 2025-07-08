import logging
import os
import time
import pickle
import json
import numpy as np
import networkx as nx
import torch
from torch import distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader

from utils.attribute_code import attribute_converter, reshape_node_attr_vec_to_mat
from utils.graph_utils import pad_adjs
from utils.visual_utils import plot_graphs_list


def load_data(config, dist_helper, eval_mode=False):
    """
    Setup training/validation/testing dataloader.
    """

    batch_size = config.test.batch_size if eval_mode else config.train.batch_size

    def _build_dataloader(in_dataset):
        if dist_helper.is_ddp:
            sampler = DistributedSampler(in_dataset)
            batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
            data_loader = DataLoader(in_dataset, sampler=sampler, batch_size=batch_size_per_gpu,
                                     pin_memory=True, num_workers=min(6, os.cpu_count()))
        else:
            data_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True,
                                     pin_memory=True, num_workers=min(6, os.cpu_count()))
        return data_loader

    # scene graph data
    train_dataset, test_dataset, train_pkl_data, test_pkl_data, train_triplet_dict, test_triplet_dict, bbox_area_stat, bbox_freq_stat, idx_to_word = load_dataset_sg(config, eval_mode=eval_mode)

    train_dl = _build_dataloader(train_dataset)
    test_dl = _build_dataloader(test_dataset)

    logging.info("Training / testing set size: {:d} / {:d}".format(len(train_dataset), len(test_dataset)))
    logging.info("Training / testing dataloader batch size: {:d} / {:d}".format(
        train_dl.batch_size, test_dl.batch_size))

    # attach additional information to the dataloader
    test_dl.pkl_data = test_pkl_data
    test_dl.idx_to_word = idx_to_word
    test_dl.train_triplet_dict = train_triplet_dict
    test_dl.test_triplet_dict = test_triplet_dict
    test_dl.bbox_area_stat = bbox_area_stat
    test_dl.bbox_freq_stat = bbox_freq_stat

    return train_dl, test_dl


def load_dataset_sg(config, eval_mode=False):
    """
    Setup training/validation/testing dataloader for scene graph datasets.
    """
    logging.info("Loading scene graph dataset...")
    time_start = time.time()

    def _build_tensors_from_pickle(pkl_path, num_node_attr_type, num_edge_attr_type, flag_node_only=False, flag_binary_edge=False):
        """
        Build scene graph dataset tensors from the pickle file.
        """

        """load dataset and initialization"""
        assert os.path.exists(pkl_path)
        data = pickle.load(open(pkl_path, 'rb'))
        pkl_data = data

        # DEBUG mode, select a subset of the dataset
        if config.dataset.subset is not None:
            assert isinstance(config.dataset.subset, int)
            set_size = config.dataset.subset

            # DEBUG: select subset based on number of nodes #
            # num_nodes = np.array([len(item['node_labels']) for item in data])
            # num_unique_nodes = np.array([len(np.unique(item['node_labels'])) for item in data])
            # num_edges = np.array([np.sum(item['edge_map'] > 0) for item in data])
            # num_unique_edges = np.array([len(np.unique(item['edge_map'])) for item in data])

            # target_num_nodes = 3
            # idx_fix_num_nodes = np.where(np.logical_and(num_nodes == target_num_nodes, num_unique_nodes == target_num_nodes))[0]  # require 4 nodes

            # flag_connected = [nx.is_connected(nx.from_numpy_array(data[i]['edge_map'])) for i in idx_fix_num_nodes]
            # flag_unique_edges = [num_unique_edges[i] == num_edges[i] + 1 for i in idx_fix_num_nodes]
            # flag_mst = [num_nodes[i] == num_edges[i] + 0 for i in idx_fix_num_nodes]
            # flag_mst = np.logical_and(flag_mst, flag_unique_edges)
            # flag_selected = np.logical_and(flag_connected, flag_mst)
            # assert np.sum(flag_selected), "No valid scene graph found!"
            # data = [data[i] for i in idx_fix_num_nodes[flag_selected]]
            # DEBUG #

            data = data[:set_size]
            logging.info("Scene graph dataset subset selection: the first {:d} data points are used".format(set_size))

        node_ls = [item['node_labels'] for item in data]  # list of numpy arrays
        if flag_node_only:
            adj_ls = None
        else:
            if 'edge_map' in data[0].keys():
                # typical case where the scene graph data has edge attributes
                adj_ls = [item['edge_map'] for item in data]  # list of numpy arrays
            else:
                adj_ls = [np.zeros([len(item['node_labels']), len(item['node_labels'])]) for item in data]  # list of numpy arrays
        if 'node_bboxes_xcyc' in data[0].keys():
            node_bbox_ls = [item['node_bboxes_xcyc'] for item in data]  # list of numpy arrays
        else:
            node_bbox_ls = None
        if 'image_id' in data[0].keys():
            image_id_ls = [item['image_id'] for item in data]
        else:
            image_id_ls = [-1] * len(data)
        image_id_out = torch.tensor(np.array(image_id_ls), dtype=torch.int64)
        config_max_node_num = config.dataset.max_node_num  # nominal max node number
        true_max_node_num = np.max([len(node) for node in node_ls])  # actual max node number

        # sanity check: verify the number of nodes
        if not flag_node_only:
            # for node-edge generation, the actual max node number <= nominal max node number
            assert true_max_node_num <= config_max_node_num
        else:
            # for node-only generation, we reshape the node attributes into adj-like tensors,
            # so we need to satisfy: the actual max node number <= square of nominal max node number
            assert np.ceil(np.sqrt(true_max_node_num)) <= config_max_node_num

        # sanity check: verify that there is no self-loop
        if adj_ls is not None:
            for adj in adj_ls:
                _diag = np.unique(np.diag(adj))
                assert len(_diag) == 1 and _diag == 0

        """pad nodes and adjs"""
        node_pad_ls, adj_pad_ls, node_flag_ls = [], [], []
        node_bbox_pad_ls = [] if node_bbox_ls is not None else None
        pad_length = config_max_node_num ** 2 if flag_node_only else config_max_node_num
        for i in range(len(node_ls)):
            node = node_ls[i]
            # node type attributes: we have at most M types and the range is in [0, M-1], e.g., M=150 for visual genome
            # note: we *do not* use 0 for padding, unlike the edge attributes
            _len_x = len(node)
            node = np.pad(node, (0, pad_length - len(node)), 'constant', constant_values=0).astype(np.float32)  # [N]
            node_flag = np.zeros_like(node)  # [N]
            node_flag[:_len_x] = 1.0  # [N]
            node_pad_ls.append(node)
            node_flag_ls.append(node_flag)

            # edge attributes: we have at most K semantic types and 1 null type, the range is in [0, K]
            # note: we use 0 for edge padding
            if adj_ls is not None:
                adj = adj_ls[i]
                adj = np.pad(adj, ((0, pad_length - len(adj)), (0, pad_length - len(adj))), 'constant', constant_values=0).astype(np.float32)
                if flag_binary_edge:
                    # binary edge attributes
                    adj = np.where(adj > 0, 1, 0)
                adj_pad_ls.append(adj)
            else:
                adj_pad_ls.append(np.zeros([pad_length, pad_length]))

            # node bounding box attributes: there are 4 attributes per node
            # (x_center, y_center, width, height), normalized to the range [0, 1]
            if node_bbox_ls is not None:
                node_bbox = node_bbox_ls[i]  # [N, 4], N = # of nodes
                assert (0 <= node_bbox).all() and (node_bbox <= 1).all()
                assert len(node_bbox) == _len_x
                node_bbox = (node_bbox - 0.5) * 2  # shift to the range of [-1, 1]
                node_bbox = np.pad(node_bbox, ((0, pad_length - len(node_bbox)), (0, 0)), 'constant', constant_values=0)
                node_bbox_pad_ls.append(node_bbox)

        adj_pad = torch.tensor(np.array(adj_pad_ls), dtype=torch.float32)
        node_pad = torch.tensor(np.array(node_pad_ls), dtype=torch.float32)
        node_flags = torch.tensor(np.array(node_flag_ls)).bool()
        _node_flags = node_flags
        # at this stage, node and adj entries are in [0, 1, 2, ...]
        # as adj, zero-value could mean either padding or null-type, we must keep the node_flags

        """encode node and edge attributes"""
        node_encoding = config.train.node_encoding
        edge_encoding = config.train.edge_encoding
        assert node_encoding in ['one_hot', 'softmax', 'ddpm', 'bits']
        assert edge_encoding in ['one_hot', 'softmax', 'ddpm', 'bits']

        if node_encoding == 'one_hot':
            # defer one_hot encoding in the runner for mini-batch processing to save memory
            node_out = node_pad
        else:
            node_out = attribute_converter(node_pad, node_flags, in_encoding='int', out_encoding=node_encoding,
                                           num_attr_type=num_node_attr_type, flag_nodes=True,
                                           flag_in_ddpm_range=False, flag_out_ddpm_range=True)

        if not flag_node_only:
            if edge_encoding == 'one_hot':
                # defer one_hot encoding in the runner for mini-batch processing to save memory
                adj_out = adj_pad
            else:
                adj_out = attribute_converter(adj_pad, node_flags, in_encoding='int', out_encoding=edge_encoding,
                                              num_attr_type=num_edge_attr_type if not flag_binary_edge else 2, flag_adjs=True,
                                              flag_in_ddpm_range=False, flag_out_ddpm_range=True)
        else:
            adj_out = adj_pad

        # special case of node only generation, we treat the nodes as adjs!
        if flag_node_only:
            # reshape node attributes into adj-like tensors
            # adj_out: [B, N, N] or [B, C, N, N], adj_flags: [B, N, N]
            adj_out, adj_flags = reshape_node_attr_vec_to_mat(node_out, node_flags, matrix_size=config_max_node_num)

            # dummy node attributes
            node_out = torch.zeros_like(node_out).float()[:, :config_max_node_num]  # [B, N]
            if len(node_out.shape) == 3:
                node_out = node_out[:, :, 0]  # [B, N] <- [B, N, C]

            # rewrite the node flags
            node_flags_out = adj_flags
        else:
            node_flags_out = node_flags

        # concatenate additional node attributes if needed
        if node_bbox_pad_ls is not None:
            node_bbox_ = torch.tensor(np.stack(node_bbox_pad_ls), dtype=torch.float32)  # [B, N, 4]
            if not flag_node_only:
                # attach node bbox to the node attributes
                if node_encoding == 'one_hot':
                    # defer one_hot encoding in the runner for mini-batch processing to save memory
                    node_out = torch.cat([node_out.unsqueeze(-1), node_bbox_], dim=-1)  # [B, N, C+4]
                elif node_encoding == 'bits':
                    node_out = torch.cat([node_out, node_bbox_], dim=-1)  # [B, N, C+4]
                elif node_encoding == 'ddpm':
                    node_out = torch.cat([node_out.unsqueeze(-1), node_bbox_], dim=-1)  # [B, N, 1+4]
                else:
                    raise NotImplementedError
            else:
                # attach node bbox to the adj attributes
                adj_bbox_out, _ = reshape_node_attr_vec_to_mat(node_bbox_, _node_flags,
                                                               matrix_size=config_max_node_num)  # [B, 4, N, N]
                # attach node bbox to the node attributes
                if node_encoding == 'one_hot':
                    # one_hot encoding for flag_node_only is not supported
                    raise NotImplementedError
                elif node_encoding == 'bits':
                    adj_out = torch.cat([adj_out, adj_bbox_out], dim=1)  # [B, C+4, N, N]
                elif node_encoding == 'ddpm':
                    adj_out = torch.cat([adj_out.unsqueeze(1), adj_bbox_out], dim=1)  # [B, 1+4, N, N]
                else:
                    raise NotImplementedError
        # special case of binary edge generation, we treat the edge type as binary
        if flag_binary_edge:
            if len(adj_out.shape) == 4:
                assert adj_out.size(1) == 1
                adj_out = adj_out[:, 0]  # [B, N, N] <- [B, 1, N, N]
            else:
                pass

        # for scene graph dataset, we always keep the node flags
        dataset = TensorDataset(adj_out, node_out, node_flags_out, image_id_out)

        return dataset, pkl_data

    # read raw data
    _flag_node_only = config.train.node_only
    _flag_binary_edge = config.train.binary_edge
    config_dataset_name = config.dataset.name
    if 'test_pkl' in config.test:
        test_pkl_path = config.test.test_pkl
    else:
        test_pkl_path = None

    if 'visual_genome' in config_dataset_name:
        num_node_type, num_edge_type = 150, 51
        if test_pkl_path is not None:
            if 'layout2img' in test_pkl_path:
                num_node_type = 151  # to account for unknown type
        train_pkl_path = os.path.join('data_scenegraph/visual_genome/training_data_bbox_dbox32_np.pkl')
        test_pkl_path = os.path.join('data_scenegraph/visual_genome/validation_data_bbox_dbox32_np.pkl')
        
        train_sg_stats = pickle.load(open(os.path.join('data_scenegraph/visual_genome', 'training_data_bbox_area_stats.pkl'), 'rb'))
        test_sg_stats = pickle.load(open(os.path.join('data_scenegraph/visual_genome', 'validation_data_bbox_area_stats.pkl'), 'rb'))
        train_triplet_dict = train_sg_stats['triplet_dict_sorted']  # a dictionary, key is the triplet, value is the frequency
        test_triplet_dict = test_sg_stats['triplet_dict_sorted']  # a dictionary, key is the triplet, value is the frequency

        idx_to_word = pickle.load(open(os.path.join('data_scenegraph/visual_genome', 'idx_to_word.pkl'), 'rb'))
    elif 'coco_stuff' in config_dataset_name:
        num_node_type, num_edge_type = 171, 7
        train_pkl_path = os.path.join('data_scenegraph/coco_stuff/coco_blt_training_data_dbox32_np.pkl')
        test_pkl_path = os.path.join('data_scenegraph/coco_stuff/coco_blt_validation_data_dbox32_np.pkl')

        train_sg_stats = pickle.load(open(os.path.join('data_scenegraph/coco_stuff', 'coco_blt_training_data_area_stats.pkl'), 'rb'))
        test_sg_stats = pickle.load(open(os.path.join('data_scenegraph/coco_stuff', 'coco_blt_validation_data_area_stats.pkl'), 'rb'))
        train_triplet_dict = dict(zip(train_sg_stats['triplet_key_sorted_list'], train_sg_stats['triplet_value_sorted_list']))   # a dictionary, key is the triplet, value is the frequency
        test_triplet_dict = dict(zip(test_sg_stats['triplet_key_sorted_list'], test_sg_stats['triplet_value_sorted_list']))   # a dictionary, key is the triplet, value is the frequency

        idx_to_word = pickle.load(open(os.path.join('data_scenegraph/coco_stuff', 'idx_to_word.pkl'), 'rb'))
    else:
        raise NotImplementedError
    
    # load bounding box area statistics from evaluation dataset
    if 'node_bbox_area_avg_dict_sorted' in test_sg_stats.keys():
        bbox_area_stat = test_sg_stats['node_bbox_area_avg_dict_sorted']     # [num_node_types]
    elif 'node_bbox_area_avg_key_sorted_list' in test_sg_stats and 'node_bbox_area_avg_value_sorted_list' in test_sg_stats:
        _keys = test_sg_stats['node_bbox_area_avg_key_sorted_list']
        _values = test_sg_stats['node_bbox_area_avg_value_sorted_list']
        bbox_area_stat = dict(zip(_keys, _values))
    else:
        raise ValueError("No bbox area stat found in the stats file")

    # load bounding box frequency statistics
    if 'node_dict_sorted' in test_sg_stats.keys():
        bbox_freq_stat = test_sg_stats['node_dict_sorted']                   # [num_node_types]
    elif 'node_key_sorted_list' in test_sg_stats and 'node_value_sorted_list' in test_sg_stats:
        _keys = test_sg_stats['node_key_sorted_list']
        _values = test_sg_stats['node_value_sorted_list']
        bbox_freq_stat = dict(zip(_keys, _values))
    else:
        raise ValueError("No bbox freq stat found in the stats file")
    
    assert os.path.exists(train_pkl_path) and os.path.exists(test_pkl_path)

    if config.dataset.subset is not None:
        # in subset mode, we let test dataset to be the same as train dataset to evaluate overfitting performance
        train_dataset, train_pkl_data = _build_tensors_from_pickle(train_pkl_path, num_node_type, num_edge_type, _flag_node_only, _flag_binary_edge)
        test_dataset, test_pkl_data = train_dataset, train_pkl_data
    else:
        # normal loading
        test_dataset, test_pkl_data = _build_tensors_from_pickle(test_pkl_path, num_node_type, num_edge_type, _flag_node_only, _flag_binary_edge)
        if eval_mode:
            train_dataset, train_pkl_data = test_dataset, test_pkl_data
        else:
            train_dataset, train_pkl_data = _build_tensors_from_pickle(train_pkl_path, num_node_type, num_edge_type, _flag_node_only, _flag_binary_edge)

    time_spent = time.time() - time_start
    logging.info("Scene graph dataset loaded, time: {:.2f}".format(time_spent))
    return train_dataset, test_dataset, train_pkl_data, test_pkl_data, train_triplet_dict, test_triplet_dict, \
           bbox_area_stat, bbox_freq_stat, idx_to_word

