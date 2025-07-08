import argparse
import os
import sys
import pickle
import time
import numpy as np
import torch
from torchvision.ops import box_convert
import matplotlib.pyplot as plt

PROJ_DIR = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, PROJ_DIR)
from utils.graph_utils import mask_nodes
from utils.visual_utils import plot_scene_graph_bbox
from evaluation.bbox_metrics import SceneGraphEvaluator


def parse_arguments():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="Running Experiments")

    # general options
    parser.add_argument('--npz', type=str, required=True, help="Path to the generated data.")
    parser.add_argument('--train_set', action='store_true', help='Whether to load the full training set.')

    args = parser.parse_args()

    assert os.path.exists(args.npz)
    print(args)

    return args


def eval_blt_ioa(pred_bbox, gt_bbox, node_flags, full_gt_x_bbox, full_node_flags, flag_train_set, save_dir):
    """BLT iou metrics"""
    # compute the perceptual iou
    canvas_size = 32
    eval_helper = SceneGraphEvaluator()

    pred_bbox_blt_iou_percp_ls = eval_helper.compute_bbox_ioa(pred_bbox, node_flags, canvas_size, flag_perceptual_iou=True)
    gt_bbox_blt_iou_percp_ls = eval_helper.compute_bbox_ioa(gt_bbox, node_flags, canvas_size, flag_perceptual_iou=True)

    if flag_train_set:
        full_gt_bbox = box_convert(full_gt_x_bbox, in_fmt='cxcywh', out_fmt='xyxy').clip(min=0.0, max=1.0)  # [X, N, 4]
        full_gt_bbox_blt_iou_percp_ls = eval_helper.compute_bbox_ioa(full_gt_bbox, full_node_flags, canvas_size, flag_perceptual_iou=True)
        print("BLT perceptual IoU metrics: full_gt: {:.4f}".format(np.mean(full_gt_bbox_blt_iou_percp_ls)))

    # show the perceptual iou distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(pred_bbox_blt_iou_percp_ls, bins=100, alpha=0.5, label='pred')
    ax.hist(gt_bbox_blt_iou_percp_ls, bins=100, alpha=0.5, label='gt')
    # if args.train_set:
    #     ax.hist(full_gt_bbox_blt_iou_percp_ls, bins=100, alpha=0.5, label='full_gt')
    ax.set_xlabel('perceptual iou')
    ax.set_ylabel('frequency')
    ax.set_title("Mean: pred: {:.4f}, gt: {:.4f}. Canvas size: {:d}".format(np.mean(pred_bbox_blt_iou_percp_ls), np.mean(gt_bbox_blt_iou_percp_ls), canvas_size))
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'hist_perceptual_iou_canvas_{:d}.png'.format(canvas_size)), dpi=300, bbox_inches='tight')

    print("BLT perceptual IoU metrics: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_bbox_blt_iou_percp_ls), np.mean(gt_bbox_blt_iou_percp_ls)))

    # compare the effect of canvas size on perceptual iou
    # canvas_size_ls = [32, 64, 128, 224, 384, 512, 1024]
    # pred_blt_iou_avg_ls, gt_blt_iou_avg_ls = [], []
    # for canvas_size in canvas_size_ls:
    #     pred_bbox_blt_iou_percp_ls = _compute_blt_iou(pred_bbox, node_flags, canvas_size=canvas_size)
    #     gt_bbox_blt_iou_percp_ls = _compute_blt_iou(gt_bbox, node_flags, canvas_size=canvas_size)
    #     pred_blt_iou_avg_ls.append(np.mean(pred_bbox_blt_iou_percp_ls))
    #     gt_blt_iou_avg_ls.append(np.mean(gt_bbox_blt_iou_percp_ls))
    #
    # # plot
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(canvas_size_ls, pred_blt_iou_avg_ls, '.-', label='pred')
    # ax.plot(canvas_size_ls, gt_blt_iou_avg_ls, '.-', label='gt')
    # ax.set_xlabel('canvas size')
    # ax.set_ylabel('perceptual iou')
    # ax.legend()
    # plt.savefig(os.path.join(save_dir, 'perceptual_iou_vs_canvas_size.png'), dpi=300, bbox_inches='tight')

    # compute the vanilla iou
    pred_blt_v_iou_ls = eval_helper.compute_bbox_ioa(pred_bbox, node_flags, canvas_size, flag_vanilla_iou=True)
    gt_blt_v_iou_ls = eval_helper.compute_bbox_ioa(gt_bbox, node_flags, canvas_size, flag_vanilla_iou=True)

    if flag_train_set:
        full_gt_blt_v_iou_ls = eval_helper.compute_bbox_ioa(full_gt_bbox, full_node_flags, canvas_size, flag_vanilla_iou=True)
        print("BLT vanilla IoU metrics: full_gt: {:.4f}".format(np.mean(full_gt_blt_v_iou_ls)))

    # show the vanilla iou distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(pred_blt_v_iou_ls, bins=100, alpha=0.5, label='pred')
    ax.hist(gt_blt_v_iou_ls, bins=100, alpha=0.5, label='gt')
    # if args.train_set:
    #     ax.hist(full_gt_bbox_blt_iou_percp_ls, bins=100, alpha=0.5, label='full_gt')
    ax.set_xlabel('vanilla iou')
    ax.set_ylabel('frequency')
    ax.set_title("Mean: pred: {:.4f}, gt: {:.4f}.".format(np.mean(pred_blt_v_iou_ls), np.mean(gt_blt_v_iou_ls)))
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'hist_vanilla_iou.png'), dpi=300, bbox_inches='tight')

    print("BLT vanilla IoU metrics: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_blt_v_iou_ls), np.mean(gt_blt_v_iou_ls)))

    """BLT overlap metrics"""
    pred_blt_overlap_ls = eval_helper.compute_bbox_ioa(pred_bbox, node_flags, canvas_size, flag_overlap=True)
    gt_blt_overlap_ls = eval_helper.compute_bbox_ioa(gt_bbox, node_flags, canvas_size, flag_overlap=True)
    if flag_train_set:
        full_gt_blt_overlap_ls = eval_helper.compute_bbox_ioa(full_gt_bbox, full_node_flags, canvas_size, flag_overlap=True)
        print("BLT overlap metrics: full_gt: {:.4f}".format(np.mean(full_gt_blt_overlap_ls)))

    # show the overlap distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(pred_blt_overlap_ls, bins=100, alpha=0.5, label='pred')
    ax.hist(gt_blt_overlap_ls, bins=100, alpha=0.5, label='gt')
    ax.set_xlabel('overlap')
    ax.set_ylabel('frequency')
    ax.set_title("Mean: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_blt_overlap_ls), np.mean(gt_blt_overlap_ls)))
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'hist_overlap.png'), dpi=300, bbox_inches='tight')

    print("BLT overlap metrics: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_blt_overlap_ls), np.mean(gt_blt_overlap_ls)))

    """BLT alignment metrics"""
    pred_blt_alignment_ls = eval_helper.compute_bbox_ioa(pred_bbox, node_flags, canvas_size, flag_alignment=True)
    gt_blt_alignment_ls = eval_helper.compute_bbox_ioa(gt_bbox, node_flags, canvas_size, flag_alignment=True)
    if flag_train_set:
        full_gt_blt_alignment_ls = eval_helper.compute_bbox_ioa(full_gt_bbox, full_node_flags, canvas_size, flag_alignment=True)
        print("BLT alignment metrics: full_gt: {:.4f}".format(np.mean(full_gt_blt_alignment_ls)))

    # show the alignment distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(pred_blt_alignment_ls, bins=100, alpha=0.5, label='pred')
    ax.hist(gt_blt_alignment_ls, bins=100, alpha=0.5, label='gt')
    ax.set_xlabel('alignment')
    ax.set_ylabel('frequency')
    ax.set_title("Mean: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_blt_alignment_ls), np.mean(gt_blt_alignment_ls)))
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'hist_alignment.png'), dpi=300, bbox_inches='tight')

    print("BLT alignment metrics: pred: {:.4f}, gt: {:.4f}".format(np.mean(pred_blt_alignment_ls), np.mean(gt_blt_alignment_ls)))


def load_gt_data(dataset_nm, flag_load_train_set):
    if 'visual_genome' in dataset_nm:
        val_sg_stats = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'visual_genome', 'validation_data_bbox_area_stats.pkl'), 'rb'))
        idx_to_word = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph/visual_genome', 'idx_to_word.pkl'), 'rb'))
        val_triplet_dict = val_sg_stats['triplet_dict_sorted']  # a dictionary, key is the triplet, value is the frequency
        train_sg_stats = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'visual_genome', 'training_data_bbox_area_stats.pkl'), 'rb'))
        train_triplet_dict = train_sg_stats['triplet_dict_sorted']
    elif 'coco_stuff' in dataset_nm:
        val_sg_stats = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'coco_stuff', 'coco_blt_validation_data_area_stats.pkl'), 'rb'))
        idx_to_word = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph/coco_stuff', 'idx_to_word.pkl'), 'rb'))
        val_triplet_dict = dict(zip(val_sg_stats['triplet_key_sorted_list'], val_sg_stats['triplet_value_sorted_list']))   # a dictionary, key is the triplet, value is the frequency
        train_sg_stats = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'coco_stuff', 'coco_blt_training_data_area_stats.pkl'), 'rb'))
        train_triplet_dict = dict(zip(train_sg_stats['triplet_key_sorted_list'], train_sg_stats['triplet_value_sorted_list']))
    else:
        raise NotImplementedError

    if 'node_bbox_area_avg_dict_sorted' in val_sg_stats.keys():
        bbox_area_stat = val_sg_stats['node_bbox_area_avg_dict_sorted']     # [num_node_types]
    elif 'node_bbox_area_avg_key_sorted_list' in val_sg_stats and 'node_bbox_area_avg_value_sorted_list' in val_sg_stats:
        _keys = val_sg_stats['node_bbox_area_avg_key_sorted_list']
        _values = val_sg_stats['node_bbox_area_avg_value_sorted_list']
        bbox_area_stat = dict(zip(_keys, _values))
    else:
        raise ValueError("No bbox area stat found in the stats file")

    if 'node_dict_sorted' in val_sg_stats.keys():
        bbox_freq_stat = val_sg_stats['node_dict_sorted']                   # [num_node_types]
    elif 'node_key_sorted_list' in val_sg_stats and 'node_value_sorted_list' in val_sg_stats:
        _keys = val_sg_stats['node_key_sorted_list']
        _values = val_sg_stats['node_value_sorted_list']
        bbox_freq_stat = dict(zip(_keys, _values))
    else:
        raise ValueError("No bbox freq stat found in the stats file")

    if flag_load_train_set:
        if 'visual_genome' in dataset_nm:
            full_gt_data = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'visual_genome', 'training_data_bbox_dbox32_np.pkl'), 'rb'))
        elif 'coco_stuff' in dataset_nm:
            full_gt_data = pickle.load(open(os.path.join(PROJ_DIR, 'data_scenegraph', 'coco_stuff', 'coco_blt_training_data_dbox32_np.pkl'), 'rb'))
        else:
            raise NotImplementedError

        assert 'node_bboxes_xcyc' in full_gt_data[0].keys()
        full_gt_x_bbox = [item['node_bboxes_xcyc'] for item in full_gt_data]  # list of numpy arrays
        pad_length = max([item.shape[0] for item in full_gt_x_bbox])
        full_node_flags = np.zeros([len(full_gt_x_bbox), pad_length], dtype=np.int32)

        for i, item in enumerate(full_gt_x_bbox):
            if item.shape[0] < pad_length:
                full_gt_x_bbox[i] = np.concatenate([item, np.zeros([pad_length - item.shape[0], 4])], axis=0)
                full_node_flags[i, :item.shape[0]] = 1
            else:
                full_node_flags[i] = 1

        full_gt_x_bbox = torch.tensor(np.stack(full_gt_x_bbox, axis=0))  # [X, N, 4], X could be super large
        full_node_flags = torch.tensor(full_node_flags).bool()  # [X, N]
        print("Successfully loaded full gt training data. Shape: {}".format(full_gt_x_bbox.shape))
    else:
        full_gt_x_bbox = torch.zeros([1, 4])  # dummy
        full_node_flags = torch.zeros([1, 1]).bool()  # dummy

    return idx_to_word, bbox_area_stat, bbox_freq_stat, val_triplet_dict, train_triplet_dict, full_gt_x_bbox, full_node_flags


def _np_array_to_torch_tensor(np_array_or_list):
    if isinstance(np_array_or_list, np.ndarray):
        return torch.tensor(np_array_or_list)
    elif isinstance(np_array_or_list, list):
        for item in np_array_or_list:
            assert isinstance(item, np.ndarray)
        return [torch.tensor(item) for item in np_array_or_list]


def _torch_tensor_to_np_array(torch_tensor_or_list):
    if isinstance(torch_tensor_or_list, torch.Tensor):
        return torch_tensor_or_list.cpu().numpy()
    elif isinstance(torch_tensor_or_list, list):
        for item in torch_tensor_or_list:
            assert isinstance(item, torch.Tensor)
        return [item.cpu().numpy() for item in torch_tensor_or_list]


def main():
    """Init section"""
    # parse arguments
    args = parse_arguments()

    # init: load gt data
    if 'visual_genome' in args.npz or '_vg' in args.npz:
        dataset_nm = 'visual_genome'
        raw_num_node_type, raw_num_adj_type, num_allowed_nodes = 150, 51, 62
    elif 'coco_stuff' in args.npz or '_coco' in args.npz:
        dataset_nm = 'coco_stuff'
        raw_num_node_type, raw_num_adj_type, num_allowed_nodes = 171, 7, 33
    else:
        raise NotImplementedError

    idx_to_word, bbox_area_stat, bbox_freq_stat, val_triplet_dict, train_triplet_dict, full_gt_x_bbox, full_node_flags = load_gt_data(dataset_nm, flag_load_train_set=True)

    # init: load generated samples
    data = np.load(args.npz)
    samples_x, samples_a, gt_x, gt_a = data['samples_x'], data['samples_a'], data['gt_x'], data['gt_a']
    samples_x_bbox, gt_x_bbox, node_flags = data['samples_x_bbox'], data['gt_x_bbox'], data['gt_node_flags']

    samples_x, samples_a, gt_x, gt_a = _np_array_to_torch_tensor([samples_x, samples_a, gt_x, gt_a])
    samples_x_bbox, gt_x_bbox, node_flags = _np_array_to_torch_tensor([samples_x_bbox, gt_x_bbox, node_flags])
    
    ### DEBUG: select a subset ###
    # subset_size = 200
    # samples_x, samples_a, gt_x, gt_a = samples_x[:subset_size], samples_a[:subset_size], gt_x[:subset_size], gt_a[:subset_size]
    # samples_x_bbox, gt_x_bbox, node_flags = samples_x_bbox[:subset_size], gt_x_bbox[:subset_size], node_flags[:subset_size]
    ### DEBUG: select a subset ###

    flag_node_only = 'layoutdm' in os.path.abspath(args.npz)
    # sg_plot_path = os.path.join(PROJ_DIR, 'helper', 'eval_sg_helper_plots')
    sg_plot_path = os.path.join(os.path.dirname(args.npz), 'eval_sg_helper_plots')
    os.makedirs(sg_plot_path, exist_ok=True)

    time_start = time.time()

    # we use xyxy format throughout the bounding box evaluation process
    pred_bbox = box_convert(samples_x_bbox, in_fmt='cxcywh', out_fmt='xyxy').clip(min=0.0, max=1.0)    # [B, N, 4]
    gt_x_bbox = box_convert(gt_x_bbox, in_fmt='cxcywh', out_fmt='xyxy').clip(min=0.0, max=1.0)           # [B, N, 4]
    num_samples, num_nodes = samples_x.shape[:2]
    print("Statistics: num_samples: {:d}, num_nodes (max): {:d}".format(num_samples, num_nodes))

    """BLT IOA (iou, overlap, alignment) metrics"""
    eval_helper = SceneGraphEvaluator()
    eval_blt_ioa(pred_bbox, gt_x_bbox, node_flags, full_gt_x_bbox, full_node_flags, args.train_set, sg_plot_path)

    """MMD metrics"""
    mmd_kernels = ['gaussian']
    node_deg_mmd = eval_helper.compute_node_degree_mmd(samples_a, gt_a, mmd_kernels)
    node_type_mmd = eval_helper.compute_node_type_mmd(samples_x, gt_x, node_flags, node_flags, raw_num_node_type, mmd_kernels)
    edge_type_mmd = eval_helper.compute_edge_type_mmd(samples_a, gt_a, node_flags, node_flags, raw_num_adj_type, mmd_kernels)
    print("Node degree MMD: {}".format(node_deg_mmd))
    print("Node type MMD: {}".format(node_type_mmd))
    print("Edge type MMD: {}".format(edge_type_mmd))

    # triplet type TV
    if flag_node_only:
        print("Skip TV triplet type evaluation for node only model")
    else:
        print("{} Evaluate triplet type TV using validation set statistics {}".format("="*10, "="*10))
        triplet_to_count_val = list(val_triplet_dict.keys())
        triplet_tv_dist_rej_val, triplet_tv_dist_all_val, triplet_tv_dist_full_val, triplet_novelty_val = eval_helper.compute_triplet_tv_dist(samples_a, samples_x, node_flags, val_triplet_dict, triplet_to_count_val)
        print(f'Truncated TV distance rejecting novel triplets: {triplet_tv_dist_rej_val}')
        print(f'Truncated TV distance accepting novel triplets: {triplet_tv_dist_all_val}')
        print(f'Full TV distance considering novel and GT triplets: {triplet_tv_dist_full_val}')
        print(f'Novel triplet percentage: {triplet_novelty_val}')

        print("{} Evaluate triplet type TV using training set statistics {}".format("="*10, "="*10))
        triplet_to_count_train = list(train_triplet_dict.keys())
        triplet_tv_dist_rej_train, triplet_tv_dist_all_train, triplet_tv_dist_full_train, triplet_novelty_train = eval_helper.compute_triplet_tv_dist(samples_a, samples_x, node_flags, train_triplet_dict, triplet_to_count_train)
        print(f'Truncated TV distance rejecting novel triplets: {triplet_tv_dist_rej_train}')
        print(f'Truncated TV distance accepting novel triplets: {triplet_tv_dist_all_train}')
        print(f'Full TV distance considering novel and GT triplets: {triplet_tv_dist_full_train}')
        print(f'Novel triplet percentage: {triplet_novelty_train}')

        # import matplotlib.pyplot as plt
        # plt.bar(np.arange(len(tv_gt_triplet_hist)), label='gt', height=tv_gt_triplet_hist)
        # plt.bar(np.arange(len(tv_gt_triplet_hist)), height=tv_pred_triplet_hist, label='pred')
        # plt.legend()
        # plt.show()

    """bounding box metrics"""
    weight_by_area = np.array([bbox_area_stat[k] for k in sorted(bbox_area_stat.keys())])  # area for node type 0, 1, 2, ...
    weight_by_area = weight_by_area / np.sum(weight_by_area)  # normalize to sum to 1

    weight_by_freq = np.array([bbox_freq_stat[k] for k in sorted(bbox_freq_stat.keys())])  # freq for node type 0, 1, 2, ...
    weight_by_freq = weight_by_freq / np.sum(weight_by_freq)  # normalize to sum to 1

    print("Computing mat_f1...")
    weights_ls = [np.ones_like(weight_by_area), weight_by_area, weight_by_freq]

    mat_f1_backup = eval_helper.compute_bbox_f1(pred_bbox, samples_x, node_flags, gt_x_bbox, gt_x, node_flags, weights_ls)
    mat_f1_vanilla, mat_f1_area, mat_f1_freq = [np.squeeze(arr, axis=2) for arr in np.dsplit(mat_f1_backup, 3)]

    def _print_mat_f1_info(mat_f1, keyword=None):
        # mat_f1: [X, Y], X is number of generated samples, Y is number of gt samples
        matching_bbox_metrics = {
            'avg_max_f1': mat_f1.max(axis=-1).mean(),
            'avg_mean_f1': mat_f1.mean(axis=-1).mean(),
            'avg_median_f1': np.median(mat_f1, axis=-1).mean(),
        }
        print("{:s} F1 bbox metrics:".format(keyword if keyword is not None else ''))
        for k, v in matching_bbox_metrics.items():
            print("{}: {}".format(k, v))

    dataset_nm = 'visual_genome' if 'visual_genome' in args.npz else 'coco_stuff'

    _print_mat_f1_info(mat_f1_vanilla, keyword='Vanilla')
    plot_scene_graph_bbox(samples_x, pred_bbox, samples_a, gt_x, gt_x_bbox, gt_a,
                          mat_f1_vanilla, node_flags, node_flags, idx_to_word, save_dir=sg_plot_path, title=f'bbox_vanilla_{dataset_nm}', num_plots=10)

    _print_mat_f1_info(mat_f1_area, keyword='Area weighted')

    plot_scene_graph_bbox(samples_x, pred_bbox, samples_a, gt_x, gt_x_bbox, gt_a,
                          mat_f1_vanilla, node_flags, node_flags, idx_to_word, save_dir=sg_plot_path, title=f'bbox_area_{dataset_nm}', num_plots=10)

    _print_mat_f1_info(mat_f1_freq, keyword='Freq weighted')
    plot_scene_graph_bbox(samples_x, pred_bbox, samples_a, gt_x, gt_x_bbox, gt_a,
                          mat_f1_vanilla, node_flags, node_flags, idx_to_word, save_dir=sg_plot_path, title=f'bbox_freq_{dataset_nm}', num_plots=10)

    # compute node-type agnostic bbox metrics
    dummy_x = mask_nodes(torch.ones_like(samples_x), node_flags)
    mat_f1_no_node_type = eval_helper.compute_bbox_f1(pred_bbox, dummy_x, node_flags, gt_x_bbox, dummy_x, node_flags, class_weight_ls=None).squeeze(2)
    _print_mat_f1_info(mat_f1_no_node_type, keyword='No node type')
    plot_scene_graph_bbox(samples_x, pred_bbox, samples_a, gt_x, gt_x_bbox, gt_a,
                          mat_f1_no_node_type, node_flags, node_flags, idx_to_word, save_dir=sg_plot_path, title=f'bbox_no_type_{dataset_nm}', num_plots=10)

    time_mmd = time.time() - time_start
    print("Total time for MMD computation: {:.3f}s".format(time_mmd))

    # log key metrics to a file
    log_file = os.path.join(sg_plot_path, 'eval_metrics.txt')
    def _write_mat_f1_info(mat_f1, keyword=None, writable=None):
        # mat_f1: [X, Y], X is number of generated samples, Y is number of gt samples
        matching_bbox_metrics = {
            'avg_max_f1': mat_f1.max(axis=-1).mean(),
            'avg_mean_f1': mat_f1.mean(axis=-1).mean(),
            'avg_median_f1': np.median(mat_f1, axis=-1).mean(),
        }
        if writable is not None:
            writable.write("{:s} F1 bbox metrics:\n".format(keyword if keyword is not None else ''))
            for k, v in matching_bbox_metrics.items():
                writable.write("{}: {}\n".format(k, v))

    with open(log_file, 'w') as f:
        f.write("Below are the evaluation metrics for the generated samples stored at {}\n".format(args.npz))
        f.write("Number of generated samples: {}, Number of reference samples: {}\n".format(samples_x.shape[0], gt_x.shape[0]))
        f.write("Node degree MMD: {}\n".format(node_deg_mmd))
        f.write("Node type MMD: {}\n".format(node_type_mmd))
        f.write("Edge type MMD: {}\n".format(edge_type_mmd))
        f.write("Node type Full TV on val set: {}\n".format(triplet_tv_dist_full_val))
        f.write("Node type TV novelty on val set: {}\n".format(triplet_novelty_val))
        f.write("Node type Full TV on train set: {}\n".format(triplet_tv_dist_full_val))
        f.write("Node type TV novelty on train set: {}\n".format(triplet_novelty_train))

        _write_mat_f1_info(mat_f1_vanilla, keyword='Vanilla', writable=f)
        _write_mat_f1_info(mat_f1_area, keyword='Area weighted', writable=f)
        _write_mat_f1_info(mat_f1_freq, keyword='Freq weighted', writable=f)
        _write_mat_f1_info(mat_f1_no_node_type, keyword='No node type', writable=f)

    print("Evaluation metrics are saved at {}".format(log_file))

    breakpoint()


if __name__ == "__main__":
    main()
