import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import logging


def compute_sg_statistics(pred_data, gt_data, idx_to_word, save_path):
    # init
    reading_len = len(gt_data)
    pred_len = len(pred_data['samples_x'])
    if len(idx_to_word['ind_to_classes']) > 150:
        num_node_type = 171  # coco_stuff, 171 types
    else:
        num_node_type = 150  # visual genome, 150 types

    # generated dataset
    # compare average number of nodes, predicate
    node_num_list, edge_num_list = [], []
    # object/predicate/triplet distribution over large sample set
    node_word_dict, edge_word_dict, triplet_word_dict = {}, {}, {}

    for node_labels, edge_map, node_flags in zip(pred_data['samples_x'], pred_data['samples_a'], pred_data['samples_node_flags']):
        node_num = node_flags.sum()
        node_num_list.append(node_num)
        edge_num_list.append((edge_map > 0).sum())

        for idx_i in range(node_num):
            node_label = int(node_labels[idx_i])
            assert node_label < num_node_type
            node_key = idx_to_word['ind_to_classes'][node_label]
            if node_key not in node_word_dict:
                node_word_dict[node_key] = 1
            else:
                node_word_dict[node_key] += 1

            for idx_j in range(node_num):
                edge_label = int(edge_map[idx_i][idx_j])
                if edge_label > 0:  # if there is an edge
                    edge_key = idx_to_word['ind_to_predicates'][edge_label]
                    if edge_key not in edge_word_dict:
                        edge_word_dict[edge_key] = 1
                    else:
                        edge_word_dict[edge_key] += 1
                    triplet_key = node_key + '_' + edge_key + '_' + idx_to_word['ind_to_classes'][int(node_labels[idx_j])]
                    if triplet_key not in triplet_word_dict:
                        triplet_word_dict[triplet_key] = 1
                    else:
                        triplet_word_dict[triplet_key] += 1

    def _normalize_dict(in_dict):
        dict_norm = {}
        dict_sum = sum(in_dict.values())
        for _key in in_dict:
            dict_norm[_key] = in_dict[_key] / dict_sum
        return dict_norm

    # normalize the three dicts
    node_word_dict_norm = _normalize_dict(node_word_dict)
    edge_word_dict_norm = _normalize_dict(edge_word_dict)
    # triplet_word_dict_norm = _normalize_dict(triplet_word_dict)

    node_num_list_gen = copy.deepcopy(node_num_list)
    edge_num_list_gen = copy.deepcopy(edge_num_list)
    node_word_dict_gen = copy.deepcopy(node_word_dict)
    edge_word_dict_gen = copy.deepcopy(edge_word_dict)
    triplet_word_dict_gen = copy.deepcopy(triplet_word_dict)
    node_word_dict_norm_gen = copy.deepcopy(node_word_dict_norm)
    edge_word_dict_norm_gen = copy.deepcopy(edge_word_dict_norm)
    # triplet_word_dict_norm_gen = copy.deepcopy(triplet_word_dict_norm)

    # GT dataset
    # compare average number of nodes, predicate
    node_num_list, edge_num_list = [], []
    # object/predicate/triplet distribution over large sample set
    node_word_dict, edge_word_dict, triplet_word_dict = {}, {}, {}

    for example_data in gt_data:
        num_nodes = example_data["node_labels"].shape[0]
        node_num_list.append(num_nodes)
        if 'edge_map' in example_data:
            edge_num_list.append((example_data['edge_map'] > 0).sum())
        else:
            edge_num_list.append(0)

        for idx_i in range(num_nodes):
            node_key = idx_to_word['ind_to_classes'][example_data["node_labels"][idx_i]]
            if node_key not in node_word_dict:
                node_word_dict[node_key] = 1
            else:
                node_word_dict[node_key] += 1

            for idx_j in range(num_nodes):
                if 'edge_map' in example_data:
                    edge_label = example_data["edge_map"][idx_i][idx_j]
                else:
                    edge_label = 0
                if edge_label > 0:
                    edge_key = idx_to_word['ind_to_predicates'][edge_label]
                    if edge_key not in edge_word_dict:
                        edge_word_dict[edge_key] = 1
                    else:
                        edge_word_dict[edge_key] += 1
                    triplet_key = node_key + '_' + edge_key + '_' + idx_to_word['ind_to_classes'][example_data["node_labels"][idx_j]]
                    if triplet_key not in triplet_word_dict:
                        triplet_word_dict[triplet_key] = 1
                    else:
                        triplet_word_dict[triplet_key] += 1

    # normalize the three dicts
    node_word_dict_norm = _normalize_dict(node_word_dict)
    edge_word_dict_norm = _normalize_dict(edge_word_dict)
    # triplet_word_dict_norm = _normalize_dict(triplet_word_dict)

    node_num_list_gt = copy.deepcopy(node_num_list)
    edge_num_list_gt = copy.deepcopy(edge_num_list)
    node_word_dict_gt = copy.deepcopy(node_word_dict)
    edge_word_dict_gt = copy.deepcopy(edge_word_dict)
    triplet_word_dict_gt = copy.deepcopy(triplet_word_dict)
    # node_word_dict_norm_gt = copy.deepcopy(node_word_dict_norm)
    # edge_word_dict_norm_gt = copy.deepcopy(edge_word_dict_norm)
    # triplet_word_dict_norm_gt = copy.deepcopy(triplet_word_dict_norm)

    # compare average number of nodes, predicate
    logging.info("Total Sample Num - Generated: %.2f \t GT: %.2f" % (pred_len, reading_len))
    logging.info("Node Number Max. - Generated: %.2f \t GT: %.2f" % (np.max(node_num_list_gen), np.max(node_num_list_gt)))
    logging.info("Node Number Min. - Generated: %.2f \t GT: %.2f" % (np.min(node_num_list_gen), np.min(node_num_list_gt)))
    logging.info("Node Number Mean - Generated: %.2f \t GT: %.2f" % (np.mean(node_num_list_gen), np.mean(node_num_list_gt)))
    logging.info("Node Number Std. - Generated: %.2f \t GT: %.2f" % (np.std(node_num_list_gen), np.std(node_num_list_gt)))
    logging.info("Edge Number Max. - Generated: %.2f \t GT: %.2f" % (np.max(edge_num_list_gen), np.max(edge_num_list_gt)))
    logging.info("Edge Number Min. - Generated: %.2f \t GT: %.2f" % (np.min(edge_num_list_gen), np.min(edge_num_list_gt)))
    logging.info("Edge Number Mean - Generated: %.2f \t GT: %.2f" % (np.mean(edge_num_list_gen), np.mean(edge_num_list_gt)))
    logging.info("Edge Number Std. - Generated: %.2f \t GT: %.2f" % (np.std(edge_num_list_gen), np.std(edge_num_list_gt)))
    logging.info("#Unique Nodes    - Generated: %.2f \t GT: %.2f" % (len(node_word_dict_gen), len(node_word_dict_gt)))
    logging.info("#Unique Edges    - Generated: %.2f \t GT: %.2f" % (len(edge_word_dict_gen), len(edge_word_dict_gt)))
    logging.info("#Unique Triplet  - Generated: %.2f \t GT: %.2f" % (len(triplet_word_dict_gen), len(triplet_word_dict_gt)))
    logging.info("#Unique Trp/Smp  - Generated: %.2f \t GT: %.2f" % (len(triplet_word_dict_gen) / pred_len, len(triplet_word_dict_gt) / reading_len))

    # plot the distribution of node, edge, triplet
    node_key_list = []
    node_freq_list = []
    node_freq_list_result = []
    node_freq_list_result_diff = []

    for key in sorted(node_word_dict_norm_gen.keys()):
        if key in node_word_dict_norm.keys():
            node_key_list.append(key)
            node_freq_list.append(node_word_dict_norm[key])
            node_freq_list_result.append(node_word_dict_norm_gen[key])
            node_freq_list_result_diff.append(node_word_dict_norm_gen[key] - node_word_dict_norm[key])

    N = len(node_key_list)
    ind = np.arange(N)
    width_ = 0.45

    fig = plt.figure(figsize=(20, 70))
    subfigs = fig.subfigures(7, 1)

    # plot node frequency
    subfigs[0].subplots()
    plt.xticks(ind + width_ / 2, node_key_list, fontsize=10, rotation='vertical')
    plt.xlabel('Node Label', fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(ind, node_freq_list_result, width=width_, label='Result node frequency')
    plt.bar(ind + width_, node_freq_list, width=width_, label='Training node frequency')
    plt.legend()
    plt.ylabel('Node Frequency', fontsize=20)
    plt.title('Node Label Frequency', fontsize=20)
    fig.savefig(os.path.join(save_path, 'node_freq.png'), bbox_inches=mtransforms.Bbox([[0, 60], [20, 70]]))
    # plt.show()

    # plot node frequency difference
    subfigs[1].subplots()
    plt.xticks(fontsize=10, rotation='vertical')
    plt.xlabel('Node Label', fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(node_key_list, node_freq_list_result_diff)
    plt.ylabel('Node Frequency Difference', fontsize=20)
    plt.title('Node Label Frequency Difference: Result - Training', fontsize=20)
    plt.savefig(os.path.join(save_path, 'node_freq_diff.png'), bbox_inches=mtransforms.Bbox([[0, 50], [20, 60]]))
    # plt.show()

    edge_key_list = []
    edge_freq_list = []
    edge_freq_list_result = []
    edge_freq_list_result_diff = []

    for key in sorted(edge_word_dict_norm_gen.keys()):
        if key in edge_word_dict_norm.keys():
            edge_key_list.append(key)
            edge_freq_list.append(edge_word_dict_norm[key])
            edge_freq_list_result.append(edge_word_dict_norm_gen[key])
            edge_freq_list_result_diff.append(edge_word_dict_norm_gen[key] - edge_word_dict_norm[key])

    N = len(edge_key_list)
    ind = np.arange(N)
    width_ = 0.45

    # plot edge frequency
    subfigs[2].subplots()
    plt.xticks(ind + width_ / 2, edge_key_list, fontsize=10, rotation='vertical')
    plt.xlabel('Edge Label', fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(ind, edge_freq_list_result, width=width_, label='Result edge frequency')
    plt.bar(ind + width_, edge_freq_list, width=width_, label='Training edge frequency')
    plt.legend()
    plt.ylabel('Edge Frequency', fontsize=20)
    plt.title('Edge Label Frequency', fontsize=20)
    plt.savefig(os.path.join(save_path, 'edge_freq.png'), bbox_inches=mtransforms.Bbox([[0, 40], [20, 50]]))
    # plt.show()

    # plot edge frequency difference
    subfigs[3].subplots()
    plt.xticks(fontsize=10, rotation='vertical')
    plt.xlabel('Edge Label', fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(edge_key_list, edge_freq_list_result_diff)
    plt.ylabel('Edge Frequency Difference', fontsize=20)
    plt.title('Edge Label Frequency Difference: Result - Training', fontsize=20)
    plt.savefig(os.path.join(save_path, 'edge_freq_diff.png'), bbox_inches=mtransforms.Bbox([[0, 30], [20, 40]]))
    # plt.show()

    relation_cnt_dict = {}
    edge_percentage_list = []
    edge_num_list = []
    node_num_list = []
    node_to_edge_dict = {}

    for node_labels, edge_map in zip(pred_data['samples_x'], pred_data['samples_a']):
        num_node = (node_labels > 0).sum()
        num_relation = (edge_map > 0).sum()
        if num_node > 1:
            edge_percentage_list.append(num_relation / (num_node * (num_node - 1)))
        else:
            edge_percentage_list.append(num_relation / (num_node * num_node))
        edge_num_list.append(num_relation)
        node_num_list.append(num_node)

        if num_relation not in relation_cnt_dict:
            relation_cnt_dict[num_relation] = 1
        else:
            relation_cnt_dict[num_relation] += 1

        if num_node not in node_to_edge_dict:
            node_to_edge_dict[num_node] = [num_relation]
        else:
            node_to_edge_dict[num_node].append(num_relation)

    # show nodes statistics
    logging.info("Total number of generated scene graphs: {:d}".format(pred_len))
    logging.info("#nodes\t #img\t %img\t\t #edges_avg\t #node^2\t %edge_occupancy")
    key_sorted = []
    key2_sorted = []
    edge_num_sorted = []
    for key in sorted(node_to_edge_dict):
        key_sorted.append(key)
        str_to_print = "{:d} \t\t {:d} \t {:.2f} \t {:.2f} \t\t {:d} \t\t {:.3f}".format(
            key, len(node_to_edge_dict[key]), len(node_to_edge_dict[key]) * 100 / pred_len,
            np.mean(node_to_edge_dict[key]), key * (key - 1),
            np.mean(node_to_edge_dict[key]) / (key * (key - 1)) * 100 if key > 1 else np.mean(node_to_edge_dict[key]) / (key * key) * 100
        )
        logging.info(str_to_print)
        key2_sorted.append(key * (key - 1))
        edge_num_sorted.append(np.mean(node_to_edge_dict[key]))

    # show edges statistics
    logging.info("#edge\t #img \t %img ratio")
    # sort by number of relations, regardless of the node numbers
    for key in sorted(relation_cnt_dict):
        logging.info("{:d} \t {:d} \t {:.2f}".format(key, relation_cnt_dict[key], relation_cnt_dict[key] * 100 / pred_len))

    # show edge type statistics
    logging.info("edge_key \t %edge_gen \t %edge_gt \t %edge_diff")
    for edge_key, edge_freq_result, edge_freq, edge_freq_result_diff in zip(edge_key_list, edge_freq_list_result,
                                                                            edge_freq_list, edge_freq_list_result_diff):
        logging.info("%s \t %.2f \t\t %.2f \t\t %.2f" % (
        edge_key.ljust(12), edge_freq_result * 100, edge_freq * 100, edge_freq_result_diff * 100))

    # plot node number vs. edge number - line
    subfigs[4].subplots()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(key_sorted, edge_num_sorted, '-o')
    plt.xlabel('Node Number', fontsize=20)
    plt.ylabel('Actual Averaged Edge Number', fontsize=20)
    plt.title('Node Number vs. Edge Number', fontsize=20)
    plt.savefig(os.path.join(save_path, 'node_num_vs_edge_num_line.png'), bbox_inches=mtransforms.Bbox([[0, 20], [20, 30]]))
    # plt.show()

    # plot node number vs. edge number - scatter
    subfigs[5].subplots()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.scatter(key_sorted, edge_num_sorted, label='Actual averaged edge number')
    plt.scatter(key_sorted, key2_sorted, label='Max edge number')
    plt.legend()
    plt.xlabel('Node Number', fontsize=20)
    plt.ylabel('Edge Number', fontsize=20)
    plt.title('Node Number vs. Edge Number', fontsize=20)
    plt.savefig(os.path.join(save_path, 'node_num_vs_edge_num_scatter.png'), bbox_inches=mtransforms.Bbox([[0, 10], [20, 20]]))
    # plt.show()

    bin_list = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    edge_percentage_bin = [0] * 10
    for entry in edge_percentage_list:
        if entry <= 0.1:
            edge_percentage_bin[0] += 1
        elif entry <= 0.2:
            edge_percentage_bin[1] += 1
        elif entry <= 0.3:
            edge_percentage_bin[2] += 1
        elif entry <= 0.4:
            edge_percentage_bin[3] += 1
        elif entry <= 0.5:
            edge_percentage_bin[4] += 1
        elif entry <= 0.6:
            edge_percentage_bin[5] += 1
        elif entry <= 0.7:
            edge_percentage_bin[6] += 1
        elif entry <= 0.8:
            edge_percentage_bin[7] += 1
        elif entry <= 0.9:
            edge_percentage_bin[8] += 1
        else:
            edge_percentage_bin[9] += 1
    edge_percentage_bin_ratio = (np.array(edge_percentage_bin) * 100 / len(edge_percentage_list)).tolist()

    # show graph sparsity
    logging.info("Edge occupancy rate and image ratio:")
    logging.info('\t'.join([_bin.ljust(6) for _bin in bin_list]))
    logging.info('\t'.join(["{:.2f}".format(_ratio).ljust(6) for _ratio in edge_percentage_bin_ratio]))

    # plot edge sparsity
    subfigs[6].subplots()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar(bin_list, edge_percentage_bin_ratio)
    plt.xlabel('Edge Occupancy Rate (in %) Bin', fontsize=20)
    plt.ylabel('Image Ratio (in %) in Dataset', fontsize=20)
    plt.title('The Sparsity of the Graph', fontsize=20)
    plt.savefig(os.path.join(save_path, 'edge_sparsity.png'), bbox_inches=mtransforms.Bbox([[0, 0], [20, 10]]))

    # plt.show()
    plt.savefig(os.path.join(save_path, "generated_stats.png"))


def get_node_adj_num_type(dataset_name, flag_sg, encoding, flag_node_only=False, flag_node_bbox=True):
    """
    Get node and edge input and output channels, which is hard-coded for various datasets.
    """
    # for [i, j] entry, we concat node i and node j types and edge [i, j] type
    if flag_sg:
        # the number of node types and edge types include the padding type
        if 'visual_genome' in dataset_name:
            raw_num_node_type, raw_num_adj_type, num_allowed_nodes = 150, 51, 62
        elif 'coco_stuff' in dataset_name:
            raw_num_node_type, raw_num_adj_type, num_allowed_nodes = 171, 7, 33
        else:
            raise NotImplementedError
        
        if encoding == 'one_hot':
            num_node_type, num_adj_type = raw_num_node_type, raw_num_adj_type
        elif encoding == 'bits':
            num_node_type, num_adj_type = np.ceil(np.log2(raw_num_node_type)).astype(int), np.ceil(np.log2(raw_num_adj_type)).astype(int)
        elif encoding == 'ddpm':
            num_node_type, num_adj_type = 1, 1
        else:
            raise NotImplementedError

        if flag_node_only:
            in_chans_node  = 2
            in_chans_adj = num_node_type

            out_chans_node = 1
            out_chans_adj = num_node_type

            num_adj_type = num_node_type
            num_node_type = 1

            if flag_node_bbox:
                in_chans_adj += 4
                out_chans_adj += 4
        else:
            in_chans_node = num_node_type * 2
            in_chans_adj = num_adj_type

            out_chans_node = num_node_type
            out_chans_adj = num_adj_type

            if flag_node_bbox:
                num_node_type += 4
                in_chans_node += 4 * 2
                out_chans_node += 4
    else:
        raise NotImplementedError
    
    info = {
        'raw_num_node_type': raw_num_node_type,
        'raw_num_adj_type': raw_num_adj_type,
        'num_allowed_nodes': num_allowed_nodes,
        'num_node_type': num_node_type,
        'num_adj_type': num_adj_type,
        'in_chans_node': in_chans_node,
        'in_chans_adj': in_chans_adj,
        'out_chans_node': out_chans_node,
        'out_chans_adj': out_chans_adj,
    }
    return info
    

def get_node_adj_model_input_output_channels(config):
    """
    Get the input and output channels for the node-adj model.
    """
    dataset_name = config.dataset.name
    flag_sg = config.flag_sg
    encoding = config.train.node_encoding
    if 'node_only' in config.train:
        flag_node_only = config.train.node_only
    else:
        flag_node_only = False

    info = get_node_adj_num_type(dataset_name, flag_sg, encoding, flag_node_only=flag_node_only)

    in_chans_model = info['in_chans_node'] + info['in_chans_adj']
    out_chans_adj_model = info['out_chans_adj']
    out_chans_node_model = info['out_chans_node']

    return in_chans_model, out_chans_adj_model, out_chans_node_model
