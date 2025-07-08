"""
Based on EDP-GNN code (modified).
https://github.com/ermongroup/GraphScoreMatching
"""

import logging
import os
import pdb
import warnings
import networkx as nx
import numpy as np
import torch
from utils.nx_multi_edge import draw_networkx_multi_edge_labels

from PIL import Image, ImageDraw, ImageFont

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color': 'black',
    'linewidths': 1,
    'width': 0.5
}


def plot_graphs_list(graphs, energy=None, node_energy_list=None, title='title', max_num=16, save_dir=None):
    """
    Plot graphs of nx.Graph objects.
    """
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = np.ceil(np.sqrt(max_num)).astype('int')
    figure = plt.figure()

    for i in range(max_num):
        idx = i * (batch_size // max_num)
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'

        if node_energy_list is not None:
            node_energy = node_energy_list[idx]
            title_str += f'\n {np.std(node_energy):.1e}'
            nx.draw(G, with_labels=False, node_color=node_energy, cmap=cm.jet, **options)
        else:
            # print(nx.get_node_attributes(G, 'feature'))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title='fig', dpi=300, fig_dir='fig'):
    """
    Figure saving helper.
    """
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(save_dir, fig_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=True)
        plt.close()
    return


def plot_graphs_adj(adjs, energy=None, node_num=None, title='title', max_num=20, save_dir=None):
    """
    Plot graphs of numpy arrays or torch tensors.
    """
    if isinstance(adjs, torch.Tensor):
        adjs = adjs.cpu().numpy()
    with_labels = (adjs.shape[-1] < 10)
    batch_size = adjs.shape[0]
    max_num = min(batch_size, max_num)
    img_c = np.ceil(np.sqrt(max_num)).astype(int)
    figure = plt.figure()
    for i in range(max_num):
        # idx = i * (adjs.shape[0] // max_num)
        idx = i
        adj = adjs[idx, :, :]
        G = nx.from_numpy_matrix(adj)
        assert isinstance(G, nx.Graph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'
        ax.title.set_text(title_str)
        nx.draw(G, with_labels=with_labels, **options)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def plot_scene_graph(samples_x, samples_a, node_flags, idx_to_word, save_dir=None, title='title',
                     flag_bin_edge=False, num_plots=1):
    """
    Plot scene graphs.
    @param samples_x: [B] list of node types
    @param samples_a: [B, N, N] list of edge types
    @param node_flags: [B, N] list of node flags
    @param idx_to_word: dict of idx to word
    @param save_dir: directory to save the figure
    @param title: title of the figure
    @param flag_bin_edge: if the edge attribute is binary
    @param num_plots: number of plots to draw
    """

    result_len = len(samples_x)

    num_fig_col = 3
    num_fig_row = 2

    for i in range(num_plots):
        # first graph to draw
        vis_start = i * num_fig_col * num_fig_row
        # last graph + 1 to draw
        vis_end = (i + 1) * (num_fig_col * num_fig_row)

        if vis_end >= result_len or vis_start >= result_len:
            continue

        fig = plt.figure(figsize=(5 * num_fig_col, 5 * num_fig_row))
        subfigs = fig.subfigures(num_fig_row, num_fig_col, wspace=0.0, hspace=0.0)

        cnt = -1
        draw_cnt = -1
        for node_labels, edge_map, _node_flags in zip(samples_x[vis_start:vis_end], samples_a[vis_start:vis_end], node_flags[vis_start:vis_end]):
            cnt += 1
            num_nodes = _node_flags.sum().long()
            draw_cnt += 1

            nodes_list = [idx_to_word['ind_to_classes'][int(node_labels[node_idx])] + str(node_idx) for node_idx in range(num_nodes)]

            edges_list = []
            edge_places = np.where(edge_map)
            subj_list = edge_places[0]
            obj_list = edge_places[1]
            assert (len(subj_list) == len(obj_list))

            triplet_list = []
            for subj, obj in zip(subj_list, obj_list):
                # remove self-loop
                if subj == obj:
                    continue
                if subj > len(nodes_list) or obj > len(nodes_list):
                    pdb.set_trace()
                edges_list.append((nodes_list[subj], nodes_list[obj]))
                triplet_list.append(nodes_list[subj] + '_' + idx_to_word['ind_to_predicates'][int(edge_map[subj][obj])] + '_' + nodes_list[obj])

            # networkx draw
            G = nx.DiGraph()
            G.add_nodes_from(nodes_list)
            G.add_edges_from(edges_list)

            subfigs[min(draw_cnt // num_fig_col, num_fig_row-1)][min(draw_cnt % num_fig_col, num_fig_col-1)].subplots()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.0, hspace=0.0)

            # plt.title(str(cnt) + "/" + str(result_len) + ": ", loc='left', fontsize=20)
            pos = nx.circular_layout(G)
            nx.draw(
                G, pos, edge_color='black', width=1, linewidths=1,  # node_size=500,
                node_color='pink', alpha=0.9,
                labels={node: node for node in G.nodes()},
                font_size=15,
                arrowsize=20,
            )
            for subj, obj in zip(subj_list, obj_list):
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels={
                        (nodes_list[subj], nodes_list[obj]): idx_to_word['ind_to_predicates'][int(edge_map[subj][obj])]
                        if not flag_bin_edge else 'e'
                    },
                    # edge_labels=None,
                    # edge_labels={
                    #     (nodes_list[subj], nodes_list[obj]): 'non-type'},
                    font_color='red',
                    rotate=False,
                    font_size=15,
                )
            x_values, y_values = zip(*pos.values())
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * 0.3
            plt.xlim(x_min - x_margin, x_max + x_margin)

        _path_to_save = os.path.join(save_dir, '{:02d}_{:s}'.format(i, title))
        plt.savefig(_path_to_save, bbox_inches='tight')
        plt.close()


def plot_scene_graph_bbox(samples_x, samples_bbox, samples_a,
                          samples_x_gt, samples_bbox_gt, samples_a_gt,
                          mat_f1, node_flags, node_flags_gt, idx_to_word, 
                          save_dir=None, title='title', num_plots=1):
    """
    Plot scene graphs with bounding boxes.
    @param samples_x: [B, N]
    @param samples_bbox: [B, N, 4]
    @param samples_a: [B, N, N]
    @param samples_x_gt: [B, N]
    @param samples_bbox_gt: [B, N, 4]
    @param samples_a_gt: [B, N, N]
    @param mat_f1: [B, B]
    @param node_flags: [B, N]
    @param node_flags_gt: [B, N]
    @param idx_to_word: dict of idx to word
    @param save_dir: directory to save the figure
    @param title: title of the figure
    @param num_plots: number of plots to draw
    """

    num_graphs = len(samples_x)
    canvas_width = 400
    canvas_height = 400
    colors_per_type = [
        "Black", "Brown", "CadetBlue", "Chocolate", "Coral",
        "Crimson", "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray",
        "DarkGreen", "DarkMagenta", "DarkOliveGreen", "DarkOrange", "DarkOrchid",
        "DarkRed", "DarkSalmon", "DarkSeaGreen", "DarkSlateBlue", "DarkSlateGray",
        "DarkTurquoise", "DarkViolet", "DeepPink", "DeepSkyBlue", "DimGray",
        "DodgerBlue", "FireBrick", "ForestGreen", "GoldenRod", "Green",
        "HotPink", "IndianRed", "Indigo", "Khaki", "LightCoral",
        "LightSlateGray", "LightSteelBlue", "Maroon", "MediumBlue", "MediumSeaGreen",
        "MediumSlateBlue", "MediumVioletRed", "MidnightBlue", "Navy", "Olive",
        "OliveDrab", "OrangeRed", "Purple", "RoyalBlue", "SaddleBrown",
        "SeaGreen", "Sienna", "SlateBlue", "SteelBlue", "Teal"]

    gen_graph_plot_idx = mat_f1.max(axis=-1).argsort()[::-1]

    total_sg_counter = -1
    num_fig_row = 2
    
    for i in range(num_plots):
        idx_start = num_fig_row * i                 # default: 0, 2, 4, 6, 8, ...
        idx_end = idx_start + 1                     # default: 1, 3, 5, 7, 9, ...

        if idx_end >= num_graphs or idx_start >= num_graphs:
            continue

        sg_idx = torch.tensor([gen_graph_plot_idx[idx_start], gen_graph_plot_idx[idx_end]]).long()
        samples_x_ls = samples_x.index_select(0, sg_idx)
        samples_a_ls = samples_a.index_select(0, sg_idx)
        samples_bbox_ls = samples_bbox.index_select(0, sg_idx)
        node_flags_ls = node_flags.index_select(0, sg_idx)

        fig = plt.figure(figsize=(30, 10))
        subfigs = fig.subfigures(num_fig_row, 4)
        for i_plot_sg in range(len(sg_idx)):
            total_sg_counter += 1
            gen_graph_id = gen_graph_plot_idx[total_sg_counter]

            """plot the generated scene graph"""
            sg_nodes, sg_edges, sg_node_flags, sg_bboxes = samples_x_ls[i_plot_sg], samples_a_ls[i_plot_sg], node_flags_ls[i_plot_sg], samples_bbox_ls[i_plot_sg]
            sg_num_nodes = sg_node_flags.sum().long().item()
            sg_node_type_ls = sg_nodes[:sg_num_nodes].tolist()
            sg_node_info_ls = [idx_to_word['ind_to_classes'][int(sg_nodes[node_idx])] + str(node_idx) for node_idx in range(sg_num_nodes)]

            sg_edge_ls, sg_triplet_ls = [], []
            for subj, obj in zip(*np.where(sg_edges)):
                sg_edge_ls.append((sg_node_info_ls[subj], sg_node_info_ls[obj]))
                sg_triplet_ls.append(sg_node_info_ls[subj] + '_' + idx_to_word['ind_to_predicates'][int(sg_edges[subj][obj])] + '_' + sg_node_info_ls[obj])

            def _build_canvas_for_scene_graph():
                # plot the bboxes on a canvas
                canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
                unique_node_type_ls = list(set(sg_node_type_ls))
                for bbox_idx in range(sg_num_nodes):
                    draw = ImageDraw.Draw(canvas)
                    x1 = ((sg_bboxes[bbox_idx][0] - sg_bboxes[bbox_idx][2] / 2).clip(0, 1)) * canvas_width
                    y1 = ((sg_bboxes[bbox_idx][1] - sg_bboxes[bbox_idx][3] / 2).clip(0, 1)) * canvas_height
                    x2 = ((sg_bboxes[bbox_idx][0] + sg_bboxes[bbox_idx][2] / 2).clip(0, 1)) * canvas_width
                    y2 = ((sg_bboxes[bbox_idx][1] + sg_bboxes[bbox_idx][3] / 2).clip(0, 1)) * canvas_height
                    if x2 > x1 and y2 > y1:
                        this_color = colors_per_type[unique_node_type_ls.index(sg_node_type_ls[bbox_idx])]
                        draw.rectangle(((x1, y1), (x2, y2)), outline=this_color)
                        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=this_color)
                        font_path = os.path.join(os.path.dirname(__file__), 'Helvetica.ttf')
                        font = ImageFont.truetype(font_path, 14)
                        draw.text((x1, y1), sg_node_info_ls[bbox_idx], fill="white", font=font)
                        del draw
                return canvas

            ax_bbox = subfigs[i_plot_sg][0].subplots()
            canvas = _build_canvas_for_scene_graph()
            ax_bbox.imshow(canvas)
            ax_bbox.set_xticks([])
            ax_bbox.set_yticks([])
            ax_bbox.set_title("Generated scene graph {:03d}/{:03d}".format(total_sg_counter, num_graphs), loc='left', fontsize=18)

            def _draw_networkx_scene_graph():
                G = nx.DiGraph()
                G.add_nodes_from(sg_node_info_ls)

                pos = nx.circular_layout(G)
                node_size = 500
                nx.draw_networkx(
                    G, pos, node_size=node_size, font_size = 12, font_color = "black",
                    node_color='pink',
                    labels={node: node for node in G.nodes()}
                )
                for subj, obj in zip(*np.where(sg_edges)):
                    G.add_edge(sg_node_info_ls[subj], sg_node_info_ls[obj], label=idx_to_word['ind_to_predicates'][int(sg_edges[subj][obj])])
                curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
                straight_edges = list(set(G.edges()) - set(curved_edges))
                nx.draw_networkx_edges(G, pos, edgelist=straight_edges, edge_color='black', width=1, node_size=node_size)
                arc_rad = 0.12
                nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', edge_color='black', width=1, node_size=node_size)

                edge_weights = nx.get_edge_attributes(G,'label')
                curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
                straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
                draw_networkx_multi_edge_labels(G, pos, edge_labels=curved_edge_labels, rotate=True, rad=arc_rad, font_color='red', font_size = 8)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=straight_edge_labels, rotate=True, font_color='red', font_size = 8)

                x_values, y_values = zip(*pos.values())
                x_max = max(x_values)
                x_min = min(x_values)
                x_margin = (x_max - x_min) * 0.3
                plt.xlim(x_min - x_margin, x_max + x_margin)

            _ = subfigs[i_plot_sg][1].subplots()
            _draw_networkx_scene_graph()

            """plot the closest retrieval results"""
            best_match = mat_f1.argmax(axis=-1)[gen_graph_id]
            f1_score = mat_f1[gen_graph_id].max()

            sg_nodes, sg_edges, sg_node_flags, sg_bboxes = samples_x_gt[best_match], samples_a_gt[best_match], node_flags_gt[best_match], samples_bbox_gt[best_match]
            sg_num_nodes = sg_node_flags.sum().long().item()
            sg_node_type_ls = sg_nodes[:sg_num_nodes].tolist()
            sg_node_info_ls = [idx_to_word['ind_to_classes'][int(sg_nodes[node_idx])] + str(node_idx) for node_idx in range(sg_num_nodes)]

            sg_edge_ls, sg_triplet_ls = [], []
            for subj, obj in zip(*np.where(sg_edges)):
                sg_edge_ls.append((sg_node_info_ls[subj], sg_node_info_ls[obj]))
                sg_triplet_ls.append(sg_node_info_ls[subj] + '_' + idx_to_word['ind_to_predicates'][int(sg_edges[subj][obj])] + '_' + sg_node_info_ls[obj])


            ax_bbox = subfigs[i_plot_sg][2].subplots()
            canvas = _build_canvas_for_scene_graph()
            ax_bbox.imshow(canvas)
            ax_bbox.set_xticks([])
            ax_bbox.set_yticks([])
            ax_bbox.set_title("Closest GT scene graph: F1: {:.3f}, ID: {:d}".format(f1_score, best_match), loc='left', fontsize=18)

            _ = subfigs[i_plot_sg][3].subplots()
            _draw_networkx_scene_graph()


        _path_to_save = os.path.join(save_dir, '{:02d}_{:s}'.format(i, title))
        plt.savefig(_path_to_save, bbox_inches='tight', dpi=150)
        plt.close()

    # plot the F1 score distribution
    plt.figure()
    ax = plt.gca()
    ax.hist(mat_f1.max(axis=-1), bins=100)
    ax.set_xlabel('Best-matching F1 score')
    ax.set_ylabel('Frequency')
    ax.set_title('F1 score distribution')
    plt.savefig(os.path.join(save_dir, 'f1_score_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

