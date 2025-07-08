import multiprocessing as mp
import time
import os
import sys
import torch
import numpy as np
from collections import Counter
from typing import Union


PROJ_DIR = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, PROJ_DIR)
from evaluation.bbox_utils import BoundingBoxes, BoundingBox, CoordinatesType, BBType, BBFormat, Evaluator, MethodAveragePrecision
from evaluation.blt_utils import get_perceptual_iou, get_average_iou, get_overlap_index, get_alignment_loss
from utils.graph_utils import mask_adjs, mask_nodes
from evaluation.stats import eval_torch_batch
from evaluation.mmd import compute_mmd, gaussian, gaussian_emd, gaussian_tv


def collect_bounding_box_per_scene(bbox_ls, type_ls, node_flag, is_gt, all_bboxes=None):
    """
    Collect bounding box for each scene graph.
    @param bbox_ls: list of bounding box for each node,         [N, 4]
    @param type_ls: list of node type for each node,            [N]
    @param node_flag: list of node flag for each node,          [N]
    @param is_gt: whether the bounding box is ground truth or not
    @param all_bboxes: BoundingBoxes object
    """
    if all_bboxes is None:
        all_bboxes = BoundingBoxes()
    for i_bbox, (bbox, node_type_id) in enumerate(zip(bbox_ls, type_ls)):
        if node_flag[i_bbox]:
            x, y, w, h = bbox
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                bb = BoundingBox(imageName=str(i_bbox), classId=node_type_id,
                                 x=x, y=y, w=w, h=h,
                                 # typeCoordinates=CoordinatesType.Relative, imgSize=(400, 400),
                                 typeCoordinates=CoordinatesType.Absolute, imgSize=None,
                                 bbType=BBType.GroundTruth if is_gt else BBType.Detected,
                                 classConfidence=None if is_gt else 1.0,
                                 format=BBFormat.XYX2Y2)
                all_bboxes.addBoundingBox(bb)
    return all_bboxes


def preprocess_bbox_batch_scenes(bbox_batch, node_type_batch, node_flag_batch, is_gt):
    """
    Preprocess the bounding box for a batch of scene graphs.
    @param bbox_batch: list of bounding box for each scene graph,   [B, N, 4]
    @param node_type_batch: list of node type for each scene graph, [B, N]
    @param node_flag_batch: list of node flag for each scene graph, [B, N]
    @param is_gt: whether the bounding box is ground truth or not
    @return: a list of BoundingBoxes object
    """
    bbox_obj_ls = []
    for _, (_bbox, _node_type, _node_flag) in enumerate(zip(bbox_batch, node_type_batch, node_flag_batch)):
        bbox_obj_per_graph = collect_bounding_box_per_scene(_bbox, _node_type, _node_flag, is_gt=is_gt, all_bboxes=None)
        bbox_obj_ls.append(bbox_obj_per_graph)
    return bbox_obj_ls


def measure_two_sets_of_bboxes(bbox_obj_1, bbox_obj_2, iou_range=np.linspace(0.05, 0.5, 10), class_weight_ls=None):
    """
    Measure the similarity between two sets of bounding boxes by using average F1 score.
    By default, bbox_obj_1 is one of the generated samples and bbox_obj_2 one of the ground truth samples.
    """
    evaluator = Evaluator()
    average_f1_by_iou_ls = []

    if len(np.intersect1d(bbox_obj_1.getClasses(), bbox_obj_2.getClasses())):
        merged_bbox_obj = bbox_obj_1.clone()
        merged_bbox_obj.extendBoundingBoxes(bbox_obj_2)

        for iou in iou_range:
            metricsPerClass = evaluator.GetPascalVOCMetrics(
                merged_bbox_obj,    # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=iou,   # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code

            precision_per_class, recall_per_class, class_id = [], [], []
            for metric in metricsPerClass:
                class_id.append(int(metric['class']))
                if metric['AP'] == 0.0 or np.isnan(metric['AP']):
                    precision_per_class.append(0.0)
                    recall_per_class.append(0.0)
                else:
                    precision_per_class.append(np.mean(metric['precision']))
                    recall_per_class.append(np.mean(metric['recall']))

            precision_per_class = np.array(precision_per_class)
            recall_per_class = np.array(recall_per_class)
            flag_invalid = np.logical_and(precision_per_class == 0.0, recall_per_class == 0.0)
            f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clip(min=1e-6)
            f1_per_class[flag_invalid] = 0.0
            f1_per_class = np.nan_to_num(f1_per_class, nan=0.0)

            if class_weight_ls is None:
                weight_per_class_ls = [np.ones_like(f1_per_class)]
            else:
                weight_per_class_ls = [class_weights[class_id] for class_weights in class_weight_ls]
            weight_per_class_norm_ls = [weight_per_class / np.sum(weight_per_class) for weight_per_class in weight_per_class_ls]
            avg_f1_on_this_iou_ls = [np.sum(f1_per_class * weight_per_class_norm) for weight_per_class_norm in weight_per_class_norm_ls]
            average_f1_by_iou_ls.append(avg_f1_on_this_iou_ls)

        average_f1_by_iou_ls = np.stack(average_f1_by_iou_ls, axis=0)   # [num_ious, num_weights]
        mean_average_f1 = np.mean(average_f1_by_iou_ls, axis=0)         # [num_weights], mean over iou thresholds
    else:
        num_weights = 1 if class_weight_ls is None else len(class_weight_ls)
        mean_average_f1 = [0.0] * num_weights
        average_f1_by_iou_ls = np.zeros((len(iou_range), num_weights))
    return mean_average_f1, average_f1_by_iou_ls


def mp_measure_bboxes(bbox_obj_1, bbox_obj_2, idx_1, idx_2, shared_list, class_weight_ls):
    """
    Multiprocessing function to measure the similarity between two sets of bounding boxes.
    """
    maf1, _ = measure_two_sets_of_bboxes(bbox_obj_1, bbox_obj_2, class_weight_ls=class_weight_ls)
    shared_list.append((idx_1, idx_2, maf1))


KERNEL_NAME_TO_FUNC = {
    'gaussian': gaussian,
    'gaussian_emd': gaussian_emd,
    'gaussian_tv': gaussian_tv
}


def retrieve_kernerls(kernel_ls):
    # retrieve a list of MMD kernels to get different results
    if not isinstance(kernel_ls, list):
        kernels = [kernel_ls]
    else:
        kernels = kernel_ls
    assert all([item in ['gaussian', 'gaussian_emd', 'gaussian_tv'] for item in kernels])
    kernels = [KERNEL_NAME_TO_FUNC[item] for item in kernels]
    return kernels


class SceneGraphEvaluator(object):
    """
    A wrapper class for evaluating generated scene graph.
    Node attributes: type, bounding box positions
    Edge attributes: type
    Metrics:
        - Node: degree MMD
        - Node: type MMD
        - Node: bounding box F1 (weighted by area, frequency or class-agnostic)
        - Node: bounding box diversity (closest retrieval based on max-F1 score)
        - Node: bounding box IOA (vanilla IoU, perceptual IoU, overlap, alignment)
        - Edge: type MMD
        - Node and edge: triplet total variation
        - Node and edge: triplet novelty

    Configurations:
        - MMD kernel: Gaussian, GaussianTV (not PSD!), etc.
        - bbox F1 class weights: the coefficients of each node type based on area, frequency and class-agnostic
        - bbox diversity: thresholding of closest retrieval
        - triplet TV and novelty: consider all triplets in the reference data or the top-k ones

    Important:
        - If the reference data is the full training set or a sub set, we use the full training set statistics.
        - If the reference data is the test/val set, we use the test/val set statistics.
        - These statistics affects how the F1 score and triplet-metrics are computed and must be consistent with the reference data.
    """
    def __init__(self) -> None:
        """
        """
        super().__init__()

        # triplet top-k parameters for computing the triplet TV and novelty
        # if isinstance(triplet_top_k, list or np.ndarray):
        #     self.triplet_top_k = np.array(triplet_top_k).tolist()
        # else:
        #     assert isinstance(triplet_top_k, int), "triplet_top_k must be a list or an integer."
        #     self.triplet_top_k = [triplet_top_k]
        # for item in self.triplet_top_k:
        #     assert isinstance(item, int), "triplet_top_k must be a list of integers! Found {}, which is {}.".format(item, type(item))

    @staticmethod
    def _get_node_type_hist(node_types: torch.Tensor, node_flags: torch.Tensor, num_node_types: int) -> list:
        """
        General utility function: get the node type histogram per scene graph.
        @param node_types:          node type tensor, [B, N]
        @param node_flags:          node flags, [B, N]
        @param num_node_types:      number of node types in range of [0, 1, 2, ..., num_node_types-1], type 0 is not padding
        """
        node_types = mask_nodes(node_types, node_flags, value=-1.0, in_place=False).float()  # [B, N], range [-1, num_x_type-1]
        node_type_hist = []
        for i in range(node_types.shape[0]):
            hist, bin_edges = torch.histogram(node_types[i], bins=num_node_types+1, range=(-1, num_node_types))
            node_type_hist.append(hist[1:].cpu().numpy())  # remove the -1.0 padded node frequency
        # note: we return unnormalized histogram
        return node_type_hist

    @staticmethod
    def _get_edge_type_hist(edge_types: torch.Tensor, node_flags: torch.Tensor, num_edge_types: int) -> list:
        """
        General utility function: get the edge type histogram per scene graph.
        @param edge_types:          edge type tensor, [B, N, N]
        @param node_flags:          node flags, [B, N]
        @param num_edge_types:      number of edge types in range of [0, 1, 2, ..., num_a_type-1], type 0 is padding
        """
        edge_types = mask_adjs(edge_types, node_flags, value=-1.0, in_place=False).float()  # [B, N, N], range [-1, num_a_type-1]
        edge_type_hist = []
        for i in range(edge_types.shape[0]):
            hist, bin_edges = torch.histogram(edge_types[i], bins=num_edge_types+1, range=(-1, num_edge_types))
            hist_ = hist[2:].cpu().numpy()  # remove the -1.0 and 0.0 edge type frequency
            if hist_.sum() > 0:
                # only collect the edge type histogram if there is at least one edge
                edge_type_hist.append(hist[2:].cpu().numpy())
        return edge_type_hist

    @staticmethod
    def _get_triplet_type_hist(edge_types: torch.Tensor, node_types: torch.Tensor, node_flags: torch.Tensor, 
                               allowed_triplet: list, reject_novel_triplet: bool):
        """
        General utility function: get the triplet type histogram per scene graph.
        @param edge_types:              edge type tensor, [B, N, N]
        @param node_types:              node type tensor, [B, N]
        @param node_flags:              node flags, [B, N]
        @param allowed_triplet:         allowed triplet, [(node_type_1, node_type_2, edge_type), ...]
        @param reject_novel_triplet:    whether to reject the novel triplet
        """
        triplet_hist = []
        max_num_novel_triplet = 0
        for i in range(edge_types.shape[0]):
            node_idx_from_to = edge_types[i].nonzero()                                          # [X, 2], int
            node_type_from_to = node_types[i][node_idx_from_to]                                 # [X, 2], int
            predicate_type = edge_types[i][edge_types[i].nonzero(as_tuple=True)].view(-1, 1)    # [X, 1], int
            samples_triplet = torch.cat([node_type_from_to, predicate_type], dim=1)             # [X, 3], int

            # count overlap
            samples_triplet_tuple_list = [tuple(x) for x in samples_triplet.tolist()]
            samples_triplet_count = Counter(samples_triplet_tuple_list)
            _overlapping_hist = []  # the order is the same as allowed_triplet
            _novel_hist = []        # the order is arbitrary (as we don't care), it stores the novel triplet in the generated scene graph

            # count frequency for the allowed triplet (e.g., seen in the training data)
            for target_triplet in allowed_triplet:
                if target_triplet in samples_triplet_count.keys():
                    _overlapping_hist.append(samples_triplet_count[target_triplet])
                else:
                    _overlapping_hist.append(0)
            
            # count frequency for the novel triplet (e.g., unseen in the training data)
            for generated_triplet in samples_triplet_count.keys():
                if generated_triplet not in allowed_triplet:
                    _novel_hist.append(samples_triplet_count[generated_triplet])

            assert len(_overlapping_hist) == len(allowed_triplet)
            assert len(_novel_hist) == len(samples_triplet_count.keys()) - (np.array(_overlapping_hist)>0).sum()
            max_num_novel_triplet = max(max_num_novel_triplet, len(_novel_hist))

            if reject_novel_triplet:
                _out_hist = _overlapping_hist
            else:
                _out_hist = _overlapping_hist + _novel_hist

            if np.sum(_out_hist) > 0.0:
                triplet_hist.append(np.array(_out_hist))
        if not reject_novel_triplet:
            # pad the novel triplet to ensure the same size
            padded_loength = max_num_novel_triplet + len(allowed_triplet)
            for i in range(len(triplet_hist)):
                if len(triplet_hist[i]) < padded_loength:
                    triplet_hist[i] = np.concatenate([triplet_hist[i], np.zeros(padded_loength - len(triplet_hist[i]))])
        return triplet_hist

    @staticmethod
    def compute_node_degree_mmd(edge_types_gen: torch.Tensor, edge_types_ref: torch.Tensor, kernel_ls: list):
        """
        Compute the node degree MMD.
        @param edge_types_gen: predicted adjacency matrix,      [B, N, N]
        @param edge_types_ref: ground truth adjacency matrix,   [B, N, N]
        @param kernel_ls: list of MMD kernel functions
        """
        mmd_kernels = retrieve_kernerls(kernel_ls)
        results = {}
        for kernel in mmd_kernels:
            result_mmd = eval_torch_batch(ref_batch=edge_types_ref, pred_batch=edge_types_gen, kernel=kernel, methods=['degree'])
            results[kernel.__name__] = result_mmd
        return results

    @staticmethod
    def compute_node_type_mmd(node_types_gen: torch.Tensor, node_types_ref: torch.Tensor, 
                              node_flags_gen: torch.Tensor, node_flags_ref: torch.Tensor, 
                              num_node_types: int, kernel_ls: list):
        """
        Compute the node type MMD.
        @param node_types_gen: predicted node types,        [B, N]
        @param node_types_ref: ground truth node types,     [B, N]
        @param node_flags_gen: node flags,                  [B, N]
        @param node_flags_ref: node flags,                  [B, N]
        @param num_node_types: number of node types in range of [0, 1, 2, ..., num_node_types-1], type 0 is not padding
        @param kernel_ls: list of MMD kernel functions
        """
        gt_node_type_hist = SceneGraphEvaluator._get_node_type_hist(node_types_ref, node_flags_ref, num_node_types=num_node_types)
        pred_node_type_hist = SceneGraphEvaluator._get_node_type_hist(node_types_gen, node_flags_gen, num_node_types=num_node_types)
        mmd_kernels = retrieve_kernerls(kernel_ls)

        assert np.sum(gt_node_type_hist) == node_flags_ref.sum()    # sanity check
        assert np.sum(pred_node_type_hist) == node_flags_gen.sum()  # sanity check
        mmd_results = {}
        for kernel in mmd_kernels:
            node_type_mmd = compute_mmd(gt_node_type_hist, pred_node_type_hist, kernel=kernel)
            mmd_results[kernel.__name__] = node_type_mmd
        return mmd_results

    @staticmethod
    def compute_edge_type_mmd(edge_types_gen: torch.Tensor, edge_types_ref: torch.Tensor, 
                              node_flags_gen: torch.Tensor, node_flags_ref: torch.Tensor, 
                              num_edge_types: int, kernel_ls: list):
        """
        Compute the edge type MMD.
        @param edge_types_gen: predicted edge types,         [B, N, N]
        @param edge_types_ref: ground truth edge types,      [B, N, N]
        @param node_flags_gen: node flags,                   [B, N]
        @param node_flags_ref: node flags,                   [B, N]
        @param num_edge_types: number of edge types in range of [0, 1, 2, ..., num_edge_types-1], type 0 is padding
        @param kernel_ls: list of MMD kernel functions
        """
        gt_edge_type_hist = SceneGraphEvaluator._get_edge_type_hist(edge_types_ref, node_flags_ref, num_edge_types=num_edge_types)
        pred_edge_type_hist = SceneGraphEvaluator._get_edge_type_hist(edge_types_gen, node_flags_gen, num_edge_types=num_edge_types)
        mmd_kernels = retrieve_kernerls(kernel_ls)

        mmd_results = {}
        if len(gt_edge_type_hist) and len(pred_edge_type_hist):
            for kernel in mmd_kernels:
                edge_type_mmd = compute_mmd(gt_edge_type_hist, pred_edge_type_hist, kernel=kernel)
                mmd_results[kernel.__name__] = edge_type_mmd
        else:
            mmd_results = {kernel.__name__: -1.0 for kernel in mmd_kernels}
        return mmd_results

    @staticmethod
    def compute_triplet_tv_dist(edge_types_gen, node_types_gen, node_flags_gen, triplet_dict, triplet_to_count):
        """
        Compute the triplet novelty.
        @param edge_types_gen: predicted adjacency matrix, [B, N, N]
        @param node_types_gen: predicted node types, [B, N]
        @param node_flags_gen: node flags, [B, N]
        @param triplet_dict: triplet dictionary, {triplet: count}
        @param triplet_to_count: the triplet to count, [N]
        """
        pred_triplet_hist_rej = SceneGraphEvaluator._get_triplet_type_hist(edge_types_gen, node_types_gen, node_flags_gen, 
                                                    allowed_triplet=triplet_dict.keys(), reject_novel_triplet=True)
        pred_triplet_hist_all = SceneGraphEvaluator._get_triplet_type_hist(edge_types_gen, node_types_gen, node_flags_gen,
                                                    allowed_triplet=triplet_dict.keys(), reject_novel_triplet=False)
        if len(pred_triplet_hist_rej):
            tv_pred_triplet_hist_rej = np.stack(pred_triplet_hist_rej, axis=0).sum(axis=0)  # [B, N] -> [N]
            tv_pred_triplet_hist_rej = tv_pred_triplet_hist_rej / np.sum(tv_pred_triplet_hist_rej)  # normalize
        else:
            tv_pred_triplet_hist_rej = np.zeros(len(triplet_dict.keys()))
        if len(pred_triplet_hist_all):
            tv_pred_triplet_hist_all = np.stack(pred_triplet_hist_all, axis=0).sum(axis=0)  # [B, N] -> [N]
            tv_pred_triplet_hist_all = tv_pred_triplet_hist_all / np.sum(tv_pred_triplet_hist_all)  # normalize
        else:
            tv_pred_triplet_hist_all = np.zeros(len(triplet_dict.keys()))

        tv_gt_triplet_hist = np.array(list(triplet_dict.values()))

        diff_rej = tv_gt_triplet_hist - tv_pred_triplet_hist_rej
        diff_all = tv_gt_triplet_hist - tv_pred_triplet_hist_all[:len(tv_gt_triplet_hist)]
        diff_full = np.concatenate([diff_all, tv_pred_triplet_hist_all[len(tv_gt_triplet_hist):]])
        triplet_tv_dist_rej = np.abs(diff_rej[:len(triplet_to_count)]).sum()
        triplet_tv_dist_all = np.abs(diff_all[:len(triplet_to_count)]).sum()
        triplet_tv_dist_full = np.abs(diff_full).sum()
        triplet_novelty = np.abs(tv_pred_triplet_hist_all[len(tv_gt_triplet_hist):]).sum()

        # another option: MMD on triplet: this is slow
        # gt_triplet_hist = _get_triplet_type_hist(final_samples_adjs_gt, final_samples_nodes_gt, test_node_flags_gt, allowed_triplet=triplet_to_count)
        # pred_triplet_hist = _get_triplet_type_hist(final_samples_adjs, final_samples_nodes, test_node_flags, allowed_triplet=triplet_to_count)
        # triplet_type_mmd = compute_mmd(gt_triplet_hist, pred_triplet_hist, kernel=gaussian_tv)
        
        return triplet_tv_dist_rej, triplet_tv_dist_all, triplet_tv_dist_full, triplet_novelty
    
    @staticmethod
    def compute_bbox_f1(node_bbox_gen: torch.Tensor, node_types_gen: torch.Tensor, node_flags_gen: torch.Tensor, 
                        node_bbox_ref: torch.Tensor, node_types_ref: torch.Tensor, node_flags_ref: torch.Tensor, 
                        class_weight_ls: Union[list, None] = None):
        """
        Compute the bounding box F1 metrics.
        @param node_bbox_gen: predicted bounding box,           [B, N, 4]
        @param node_types_gen: predicted node type,             [B, N]
        @param node_bbox_ref: ground truth bounding box,        [B, N, 4]
        @param node_types_ref: ground truth node type,          [B, N]
        @param node_flags: node flags,                          [B, N], shared by both predicted and ground truth
        @param class_weight_ls: mean F1 weight per class,       [X]
        Note: the bounding box is in the format of x1-x2-y1-y2.
        """

        def _torch_to_numpy_helper(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                return tensor
            else:
                raise NotImplementedError
        node_bbox_gen, node_types_gen, node_flags_gen, node_bbox_ref, node_types_ref, node_flags_ref = map(_torch_to_numpy_helper, [node_bbox_gen, node_types_gen, node_flags_gen, node_bbox_ref, node_types_ref, node_flags_ref])

        """preprocess the bounding box"""
        gen_bbox_objs = preprocess_bbox_batch_scenes(node_bbox_gen, node_types_gen, node_flags_gen, is_gt=False)
        ref_bbox_objs = preprocess_bbox_batch_scenes(node_bbox_ref, node_types_ref, node_flags_ref, is_gt=True)

        if class_weight_ls is not None:
            assert node_types_ref.max() <= len(class_weight_ls[0]), "The number of classes in the ground truth set is larger than the number of classes in the dataset."
            assert node_types_gen.max() <= len(class_weight_ls[0]), "The number of classes in the generated set is larger than the number of classes in the dataset."

        """use multiprocessing to speed up"""
        # multiprocessing utility
        mp_manger = mp.Manager()
        shared_list = mp_manger.list()
        mp_func_args_ls = []
        for i in range(len(gen_bbox_objs)):
            for j in range(len(ref_bbox_objs)):
                mp_func_args_ls.append((gen_bbox_objs[i], ref_bbox_objs[j], i, j, shared_list, class_weight_ls))

        time_eval = time.time()
        print("Computing bounding box F1 metrics...")

        flag_mp = True  # only turn if off for debug purpose
        if flag_mp:
            # accelerate the computation by using multiprocessing
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(mp_measure_bboxes, mp_func_args_ls, chunksize=len(mp_func_args_ls) // os.cpu_count())
        else:
            # single process
            for mp_func_args in mp_func_args_ls:
                mp_measure_bboxes(*mp_func_args)

        time_eval = time.time() - time_eval
        print("Bounding box F1 metrics computed in {:.3f} seconds.".format(time_eval))

        num_weights = 1 if class_weight_ls is None else len(class_weight_ls)
        mat_f1 = np.zeros((len(node_bbox_gen), len(node_bbox_ref), num_weights))
        for idx_1, idx_2, maf1 in shared_list:
            mat_f1[idx_1, idx_2] = maf1

        return mat_f1

    @staticmethod
    def compute_bbox_ioa(bbox_ls, node_flags, canvas_size=32,
                         flag_vanilla_iou=False, flag_perceptual_iou=False, flag_overlap=False, flag_alignment=False,
                         return_mean=False):
        """
        Compute the self-consistency metrics for bounding boxes.
        IOA: IoU, overlap, alignment

        @param bbox_ls: list of bounding boxes, [batch_size, num_nodes, 4], the last 4 dim: x1, y1, x2, y2
        @param node_flags: list of node flags,  [batch_size, num_nodes]
        @param canvas_size: canvas size, used for computing perceptual IoU
        @param flag_vanilla_iou: whether to use vanilla IoU
        @param flag_perceptual_iou: whether to use perceptual IoU
        @param flag_overlap: whether to use overlap
        @param flag_alignment: whether to use alignment
        @param return_mean: whether to return the mean value
        """
        flags = [flag_vanilla_iou, flag_perceptual_iou, flag_overlap, flag_alignment]
        assert sum(flags) == 1, "Only one flag can be True."

        if flag_vanilla_iou:
            func_metric = get_average_iou
        elif flag_perceptual_iou:
            func_metric = get_perceptual_iou
        elif flag_overlap:
            func_metric = get_overlap_index
        elif flag_alignment:
            func_metric = get_alignment_loss
        else:
            raise NotImplementedError

        metric_per_bbox_ls = []
        for i, layout in enumerate(bbox_ls):
            layout = layout[node_flags[i]]  # [X, 4], remove the padding nodes
            func_args = [layout, canvas_size] if flag_perceptual_iou else [layout]
            metric = func_metric(*func_args)
            if metric is not None:
                metric_per_bbox_ls.append(metric)
        if return_mean:
            return np.mean(metric_per_bbox_ls)
        else:
            return metric_per_bbox_ls
        