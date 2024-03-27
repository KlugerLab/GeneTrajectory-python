import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans


def select_top_genes(adata: sc.AnnData,
                     layer: str = None,
                     min_expr_percent: float = 0.01,
                     max_expr_percent: float = 0.5,
                     ) -> np.ndarray:
    """
    Narrow down the gene list for gene-gene distance computation by focusing on the top
    2000 variable genes expressed by 1% - 50% of cells.

    ```
    assay <- "RNA"
    DefaultAssay(data_S) <- assay
    data_S <- FindVariableFeatures(data_S, nfeatures = 2000)
    all_genes <- data_S@assays[[assay]]@var.features
    expr_percent <- apply(as.matrix(data_S[[assay]]@data[all_genes, ]) > 0, 1, sum)/ncol(data_S)
    genes <- all_genes[which(expr_percent > 0.01 & expr_percent < 0.5)]
    ```
    :param adata: a scanpy Anndata object
    :param min_expr_percent: minimum fraction of cells expressing the gene
    :param max_expr_percent: maximum fraction of cells expressing the gene
    :param layer: the layer with count data (e.g. 'counts', which can be created
           as `adata.layers['counts']=adata.raw.X.copy()`)
    :return: a cell-cell graph distance matrix
    """
    if layer not in adata.layers.keys():
        raise ValueError(f'Layer {layer} not found in adata. Available {adata.layers.keys()}')
    if adata.n_vars < 2000:
        raise ValueError(f'This method should be run with at least 2000 features (genes). The data has {adata.n_vars}')

    sc.pp.calculate_qc_metrics(adata, layer=layer, inplace=True)
    sc.pp.highly_variable_genes(adata, layer=layer, n_top_genes=2000, flavor='seurat_v3', inplace=True)
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


def coarse_grain(cell_embedding,
                 gene_expression,
                 graph_dist,
                 n: int = 1000,
                 cluster: np.array = None,
                 random_seed=1):
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


def coarse_grain_adata(adata,
                       graph_dist,
                       features,
                       n=1000,
                       reduction="X_dm",
                       dims=5,
                       random_seed=1):

    if reduction not in adata.obsm_keys():
        raise ValueError(f'Reduction "{reduction}" is not present. Available: {adata.obsm_keys()}')

    cell_embedding = adata.obsm[reduction][:, :dims]
    gene_expression = adata[:, features].X
    cg = coarse_grain(cell_embedding=cell_embedding,
                      gene_expression=gene_expression,
                      graph_dist=graph_dist,
                      n=n, random_seed=random_seed)
    return cg
