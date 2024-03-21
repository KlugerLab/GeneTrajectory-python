import scanpy as sc
from sklearn.metrics import pairwise_distances

from gene_trajectories.diffusion_map import diffusion_map


def run_dm(adata: sc.AnnData,
           reduction: str = "X_pca",
           k=10,
           sigma=None,
           n_components=30,
           t=1,
           dist_mat=None,
           reduction_result="X_dm",
           ) -> None:
    """
    Run Diffusion Map on a Scanpy Anndata object

    :param adata: Scanpy Anndata object
    :param reduction: Dimensionality reduction to use, default: 'X_pca'
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor.
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `K` is specified.
    :param n_components: Number of leading nontrivial eigenvectors to export
    :param t: Number of diffusion times
    :param dist_mat: Precomputed distance matrix (optional)
    :param reduction_result: Dimensionality reduction to store result, default: 'X_dm'
    """
    if dist_mat is None:
        dist_mat = pairwise_distances(adata.obsm[reduction])

    diffu_emb, _ = diffusion_map(dist_mat, k=k, sigma=sigma, n_ev=n_components + 1, t=t)
    diffu_emb = diffu_emb[:, 1:]  # remove the first eigenvector (trivial)
    adata.obsm[reduction_result] = diffu_emb
