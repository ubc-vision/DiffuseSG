"""
Based on GDSS code, EDP-GNN code, GRAN code and GraphRNN code (modified).
https://github.com/harryjo97/GDSS
https://github.com/lrjconan/GRAN
https://github.com/ermongroup/GraphScoreMatching
https://github.com/JiaxuanYou/graph-generation
"""

import concurrent.futures
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import copy

from evaluation.mmd import compute_mmd, gaussian, gaussian_emd, gaussian_tv

PRINT_TIME = True

###############################################################################

def degree_worker(nx_graph):
    """
    Helper function for parallel computing of degree distribution.
    """
    return np.array(nx.degree_histogram(nx_graph))


def degree_stats(graph_ref_list, graph_pred_list, kernel, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    Kernel: Gaussian TV by default.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param kernel: kernel function
    @param is_parallel: whether to use parallel computing
    @return: the distance between the degree distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    """
    Helper function for parallel computing of clustering coefficient distribution.
    """
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, kernel, bins=100, is_parallel=True):
    """
    Compute the distance between the clustering coefficient distributions of two unordered sets of graphs.
    Kernel: Gaussian TV by default.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param kernel: kernel function
    @param bins: number of bins for histogram
    @param is_parallel: whether to use parallel computing
    @return: the distance between the clustering coefficient distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


###############################################################################

def spectral_worker(nx_graph):
    """
    Helper function for parallel computing of spectral distribution.
    """
    eigs = eigvalsh(nx.normalized_laplacian_matrix(nx_graph).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, kernel, is_parallel=True):
    """
    Compute the distance between the spectral distributions of two unordered sets of graphs.
    Kernel: Gaussian TV by default.
    @param graph_ref_list: list of networkx graphs
    @param graph_pred_list: list of networkx graphs
    @param kernel: kernel function
    @param is_parallel: whether to use parallel computing
    @return: the distance between the spectral distributions of two unordered sets of graphs
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral mmd: ', elapsed)
    return mmd_dist


###############################################################################

def adjs_to_graphs(adjs):
    """
    Convert a list of adjacency matrices to a list of graphs.
    @param adjs: list of adjacency matrices in numpy array
    @return: list of networkx graphs
    """
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def eval_acc_lobster_graph(graph_list):
    """
    Evaluate the accuracy of a list of graphs in predicting whether a graph is a lobster graph or not.
    @param graph_list: list of networkx graphs
    @return: accuracy scalar
    """
    graph_list = [copy.deepcopy(gg) for gg in graph_list]

    count = 0
    for gg in graph_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(graph_list))


def is_lobster_graph(nx_graph):
    """
    Check if a given graph is a lobster graph or not.
    Removing leaf nodes twice:
    lobster -> caterpillar -> path
    """
    # Check if G is a tree
    if nx.is_tree(nx_graph):
        # Check if G is a path after removing leaves twice
        leaves = [n for n, d in nx_graph.degree() if d == 1]
        nx_graph.remove_nodes_from(leaves)

        leaves = [n for n, d in nx_graph.degree() if d == 1]
        nx_graph.remove_nodes_from(leaves)

        num_nodes = len(nx_graph.nodes())
        num_degree_one = [d for n, d in nx_graph.degree() if d == 1]
        num_degree_two = [d for n, d in nx_graph.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


###############################################################################
    
METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'spectral': spectral_stats,
}

KERNEL_NAME_TO_FUNC = {
    'gaussian': gaussian,
    'gaussian_emd': gaussian_emd,
    'gaussian_tv': gaussian_tv
}


def eval_graph_list(graph_ref_list, grad_pred_list, kernel=None, methods=None):
    """
    Evaluate the graph statistics given networkx graphs of reference and generated graphs.
    @param graph_ref_list: list of networkx graphs
    @param grad_pred_list: list of networkx graphs
    @param kernel: kernel function
    @param methods: list of methods to evaluate
    @return: a dictionary of results
    """
    if kernel is None:
        kernel = KERNEL_NAME_TO_FUNC['gaussian_tv']
    elif kernel in KERNEL_NAME_TO_FUNC:
        kernel = KERNEL_NAME_TO_FUNC[kernel]  # string to function
    else:
        assert kernel in KERNEL_NAME_TO_FUNC.values(), 'Invalid kernel function'
    if methods is None:
        methods = ['degree', 'cluster', 'spectral']
    print("Size of reference graphs: {:d}, size of generated graphs: {:d}".format(
        len(graph_ref_list), len(grad_pred_list)))
    results = {}
    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list, kernel=kernel, is_parallel=False)
    results['average'] = np.mean(list(results.values()))
    print(results)
    return results


def eval_torch_batch(ref_batch, pred_batch, kernel=None, methods=None):
    """
    Evaluate the graph statistics given pytorch tensors of reference and generated adjacency matrices.
    @param ref_batch: pytorch tensor of shape (batch_size, num_nodes, num_nodes)
    @param pred_batch: pytorch tensor of shape (batch_size, num_nodes, num_nodes)
    @param kernel: kernel function
    @param methods: list of methods to evaluate
    @return: dictionary of results
    """
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    grad_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, grad_pred_list, kernel=kernel, methods=methods)
    return results
