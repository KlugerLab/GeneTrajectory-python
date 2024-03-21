from typing import Union
from typing import Optional

from scipy.stats import rankdata
import numpy as np
import pandas as pd

from gene_trajectories.diffusion_map import diffusion_map


def get_gene_embedding(dist_mat: np.array,
                       k: int = 10,
                       sigma: Union[float, np.array, list] = None,
                       n_ev: int = 30,
                       t=1):
    """
    Get the diffusion embedding of genes based on the gene-gene Wasserstein distance matrix

    :param dist_mat: dist.mat matrix; Gene-gene Wasserstein distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `K` is specified
    :param n_ev: Number of leading eigenvectors to export
    :param t: Number of diffusion times
    :return:
    """
    k = min(k, dist_mat.shape[0])
    n_ev = min(n_ev + 1, dist_mat.shape[0])
    diffu_emb, eigen_vals = diffusion_map(dist_mat=dist_mat, k=k, sigma=sigma, n_ev=n_ev, t=t)
    diffu_emb = diffu_emb[:, 1:n_ev + 1]
    eigen_vals = eigen_vals[1:n_ev + 1]
    return diffu_emb, eigen_vals


def get_randow_walk_matrix(dist_mat, k: int = 10):
    """
    Convert a distance matrix into a random-walk matrix based on adaptive Gaussian kernel

    :param dist_mat: Precomputed distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :return: Random-walk matrix

    # TODO: refactor the code up to affinity_matrix_symm as it's shared with diffusion_map
    """
    assert dist_mat.shape[0] == dist_mat.shape[1]
    dists = np.nan_to_num(dist_mat, 1e-6)
    k = min(k, dist_mat.shape[0])

    sigma = np.apply_along_axis(func1d=sorted, axis=1, arr=dists)[:, k - 1]

    affinity_matrix = np.exp(-dists ** 2 / (sigma ** 2)[:, None])
    affinity_matrix_symm = (affinity_matrix + affinity_matrix.T) / 2

    normalized_vec = 1 / affinity_matrix_symm.sum(axis=1)
    affinity_matrix_norm = (affinity_matrix_symm * normalized_vec[:, None])

    return affinity_matrix_norm


def get_gene_pseudoorder(dist_mat, subset: list[int], max_id: Optional[int] = None) -> np.array:
    """
    Order genes along a given trajectory

    :param dist_mat: Gene-gene Wasserstein distance matrix (symmetric)
    :param subset: Genes in a given trajectory
    :param max_id: Index of the terminal gene
    """
    assert dist_mat.shape[0] == dist_mat.shape[1]

    emd = dist_mat[subset][:, subset]
    dm_emb, _ = diffusion_map(emd)
    pseudoorder = rankdata(dm_emb[:, 1])

    if max_id is not None and max_id in subset:
        n = len(subset)
        if 2 * pseudoorder[subset.index(max_id)] < n:
            pseudoorder = n + 1 - pseudoorder

    pseudoorder_all = np.zeros(dist_mat.shape[0])
    pseudoorder_all[subset] = pseudoorder
    return pseudoorder_all


def extract_gene_trajectory(gene_embedding,
                            dist_mat,
                            gene_names: list,
                            n,
                            t_list,
                            dims=5,
                            k=10,
                            quantile=0.02,
                            other: str = 'Other',
                            ) -> pd.DataFrame:
    """
    Extract gene trajectories

    :param gene_embedding: Gene embedding
    :param dist_mat: Gene-gene Wasserstein distance matrix (symmetric)
    :param gene_names:
    :param n: Number of gene trajectories to retrieve
    :param t_list:  Number of diffusion times to retrieve each trajectory
    :param dims: Dimensions of gene embedding to use to identify terminal genes (extrema)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param quantile: Thresholding parameter to extract genes for each trajectory. Default: 0.02
    :param other: Label for genes not in a trajectory. Default: 'Other'
    :return: A data frame indicating gene trajectories and gene ordering along each trajectory
    """
    dist_to_origin = np.sqrt((gene_embedding[:, :dims] ** 2).sum(axis=1))
    df = pd.DataFrame(gene_embedding[:, :dims], columns=[f'DM_{i + 1}' for i in range(dims)],
                      index=gene_names).assign(selected=other)
    n_genes = gene_embedding.shape[0]

    diffusion_mat = get_randow_walk_matrix(dist_mat, k=k)

    for i in range(n):
        if sum(df.selected == other) == 0:
            print("Early stop evoked!")
            print(f"{i - 1} gene trajectories retrieved.")
            break

        dist_to_origin[df.selected != other] = -np.infty

        seed_idx = np.argmax(dist_to_origin)
        print(f'Generating trajectory from {gene_names[seed_idx]}')

        seed_diffused = np.zeros(n_genes)
        seed_diffused[seed_idx] = 1

        for _ in range(t_list[i]):
            seed_diffused = diffusion_mat @ seed_diffused

        cutoff = max(seed_diffused) * quantile

        trajectory_label = f'Trajectory-{i + 1}'
        df.loc[(seed_diffused > cutoff) & (df.selected == other), 'selected'] = trajectory_label

        df[f'Pseudoorder-{i + 1}'] = get_gene_pseudoorder(dist_mat=dist_mat,
                                                          subset=list(np.where(df.selected == trajectory_label)[0]),
                                                          max_id=seed_idx)
    return df
