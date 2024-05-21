from typing import Optional

import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

from gene_trajectory.util.input_validation import validate_matrix


def select_top_genes(
        adata: sc.AnnData,
        layer: str = None,
        min_expr_percent: float = 0.01,
        max_expr_percent: float = 0.5,
        n_variable_genes: int = 2000,
) -> np.ndarray:
    """
    Narrow down the gene list for gene-gene distance computation by focusing on the top
    2000 variable genes expressed by 1% - 50% of cells.

    :param adata: a scanpy Anndata object
    :param layer: the layer with count data (e.g. 'counts', which can be created
           as `adata.layers['counts']=adata.raw.X.copy()`)
    :param min_expr_percent: minimum fraction of cells expressing the gene
    :param max_expr_percent: maximum fraction of cells expressing the gene
    :param n_variable_genes: number of variable genes to use (default: 2000)
    :return: the cell-cell graph distance matrix
    """
    if layer not in adata.layers.keys():
        raise ValueError(f'Layer {layer} not found in adata. Available {list(adata.layers.keys())}')
    if adata.n_vars < 2000:
        raise ValueError(f'This method should be run with at least 2000 features (genes). The data has {adata.n_vars}')

    sc.pp.calculate_qc_metrics(adata, layer=layer, inplace=True)
    sc.pp.highly_variable_genes(adata, layer=layer, n_top_genes=n_variable_genes, flavor='seurat_v3', inplace=True)
    expr_percent = adata.var['n_cells_by_counts'] / adata.n_obs
    genes = adata.var_names[adata.var['highly_variable'] &
                            (expr_percent > min_expr_percent) & (expr_percent < max_expr_percent)]
    genes_sorted = adata[:, genes].var.sort_values('variances_norm', ascending=False).index
    return genes_sorted.values

# Implementation notes
# - It would look simpler to do `np.count_nonzero(adata.layer[''], axis=0)` instead of running
#   calculate_qc_metrics and getting `n_cells_by_counts`.
#   Unfortunately one needs different implementations for dense/sparse matrices or to convert using
#   - adata.to_df(layer)
#   - np.asarray(np.sum(XX, axis=0).squeeze())


def coarse_grain(
        cell_embedding: np.ndarray,
        gene_expression: np.ndarray,
        graph_dist: np.ndarray,
        n: int = 1000,
        cluster: Optional[np.array] = None,
        random_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply coarse-graining to reduce the number of cells

    :param cell_embedding: the cell embedding
    :param gene_expression: the gene expression matrix
    :param graph_dist: the graph distance matrix
    :param n: number of cells to keep
    :param cluster: specify an array to use precomputed clusters. If not specified a KMeans clustering will be performed
    :param random_seed: the random seed
    :return: the updated cell embedding and gene expression matrices
    """
    validate_matrix(gene_expression, obj_name='Gene Expression Matrix', min_value=0)
    ncells, ngenes = gene_expression.shape
    validate_matrix(cell_embedding, obj_name='Cell embedding', nrows=ncells)

    if cluster is None:
        k_means = KMeans(n_clusters=n, random_state=random_seed).fit(cell_embedding)
        cluster = k_means.labels_ # noqa

    knn_membership = np.zeros((n, cell_embedding.shape[0]))
    for i, c in enumerate(cluster):
        knn_membership[c, i] = 1
    gene_expression_updated = knn_membership @ gene_expression

    knn_membership_norm = knn_membership / np.sum(knn_membership, axis=1)[:, None]
    graph_dist_updated = knn_membership_norm @ graph_dist @ knn_membership_norm.T

    return gene_expression_updated, graph_dist_updated


def coarse_grain_adata(
        adata: sc.AnnData,
        graph_dist: np.array,
        features: list[str],
        n: int = 1000,
        reduction: str = "X_dm",
        dims: int = 5,
        random_seed=1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply coarse-graining to reduce the number of cells

    :param adata: a scanpy Anndata object
    :param graph_dist: the graph distance matrix
    :param features: the features (i.e. genes) to keep
    :param n: number of cells to keep (default = 1000)
    :param reduction: the dimensional reduction (in adata.obsm) to keep
    :param dims: the number of dimensions to keep (default = 5)
    :param random_seed: the random seed
    :return: the updated cell embedding and gene expression matrices
    """
    if reduction not in adata.obsm_keys():
        raise ValueError(f'Reduction "{reduction}" is not present. Available: {adata.obsm_keys()}')

    cell_embedding = adata.obsm[reduction][:, :dims]
    gene_expression = adata[:, features].X
    cg = coarse_grain(cell_embedding=cell_embedding,
                      gene_expression=gene_expression,
                      graph_dist=graph_dist,
                      n=n, random_seed=random_seed)
    return cg
