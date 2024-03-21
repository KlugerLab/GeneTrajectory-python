import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import igraph as ig


def get_graph_distance(adata: sc.AnnData, reduction='X_dm', k=10, dims=5):
    """
    Computes the graph distance on a Scanpy object

    :param adata: a scanpy Anndata object
    :param reduction: dimensionality reduction to use, default: 'dm'
    :param k: adaptive kernel bandwidth for each point set to be the distance to its k-th nearest neighbor
    :param dims: the dimensions to use as input features for kNN graph construction
    :return: a cell-cell graph distance matrix
    """
    # Implementation note. This could also be done using scipy, although it's 3x slower
    #
    # from scipy.sparse.csgraph import shortest_path
    # ds = scipy.sparse.csgraph.shortest_path(adj, directed=False, unweighted=True)
    #
    cell_embedding = adata.obsm[reduction][:, :dims]
    knn = NearestNeighbors(n_neighbors=k + 1).fit(cell_embedding)
    neighbors = knn.kneighbors(cell_embedding, return_distance=False)
    g = ig.Graph(((nn[0], j) for nn in neighbors for j in nn[1:]), directed=False)
    if not g.is_connected():
        raise RuntimeError('The cell-cell kNN graph has disconnected components. Please increase k.')
    graph_dist_mat = np.array(g.distances())
    print("The largest graph distance is", graph_dist_mat.max())

    return graph_dist_mat
