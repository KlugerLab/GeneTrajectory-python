import logging
from typing import Union
from typing import Optional

from scipy.stats import rankdata
import numpy as np
import pandas as pd

from gene_trajectory.diffusion_map import diffusion_map, get_symmetrized_affinity_matrix
from gene_trajectory.util.input_validation import validate_matrix

logger = logging.getLogger()


def get_gene_embedding(
        dist_mat: np.array,
        k: int = 10,
        sigma: Union[float, np.array, list] = None,
        n_ev: int = 30,
        t: int = 1,
) -> tuple[np.array, np.array]:
    """
    Get the diffusion embedding of genes based on the gene-gene Wasserstein distance matrix

    :param dist_mat: dist.mat matrix; Gene-gene Wasserstein distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `K` is specified
    :param n_ev: Number of leading eigenvectors to export
    :param t: Number of diffusion times
    :return: the diffusion embedding and the eigenvalues
    """
    validate_matrix(dist_mat, square=True)

    k = min(k, dist_mat.shape[0])
    n_ev = min(n_ev + 1, dist_mat.shape[0])
    diffu_emb, eigen_vals = diffusion_map(dist_mat=dist_mat, k=k, sigma=sigma, n_ev=n_ev, t=t)
    diffu_emb = diffu_emb[:, 1:n_ev + 1]
    eigen_vals = eigen_vals[1:n_ev + 1]
    return diffu_emb, eigen_vals


def get_randow_walk_matrix(
        dist_mat: np.array,
        k: int = 10,
) -> np.array:
    """
    Convert a distance matrix into a random-walk matrix based on adaptive Gaussian kernel

    :param dist_mat: Precomputed distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :return: Random-walk matrix
    """
    validate_matrix(dist_mat, square=True)

    affinity_matrix_symm = get_symmetrized_affinity_matrix(dist_mat=dist_mat, k=k)
    normalized_vec = 1 / affinity_matrix_symm.sum(axis=1)
    affinity_matrix_norm = (affinity_matrix_symm * normalized_vec[:, None])

    return affinity_matrix_norm


def get_gene_pseudoorder(
        dist_mat: np.array,
        subset: list[int],
        max_id: Optional[int] = None,
) -> np.array:
    """
    Order genes along a given trajectory

    :param dist_mat: Gene-gene Wasserstein distance matrix (symmetric)
    :param subset: Genes in a given trajectory
    :param max_id: Index of the terminal gene
    :return: The pseudoorder
    """
    validate_matrix(dist_mat, square=True)

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


def extract_gene_trajectory(
        gene_embedding: pd.DataFrame,
        dist_mat: np.array,
        gene_names: list[str],
        t_list: Union[float, np.array, list],
        n: Optional[int] = None,
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
    :param t_list:  Number of diffusion times to retrieve each trajectory
    :param n: Number of gene trajectories to retrieve. Will be set to the length of t_list
    :param dims: Dimensions of gene embedding to use to identify terminal genes (extrema)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param quantile: Thresholding parameter to extract genes for each trajectory. Default: 0.02
    :param other: Label for genes not in a trajectory. Default: 'Other'
    :return: A data frame indicating gene trajectories and gene ordering along each trajectory
    """
    validate_matrix(dist_mat, square=True)

    if np.isscalar(t_list):
        if n is None:
            raise ValueError(f'n should be specified if t_list is a number: {t_list}')
        t_list = np.full(n, t_list)
    elif n is None:
        n = len(t_list)
    if n != len(t_list):
        raise ValueError(f't_list ({t_list}) should have the same dimension as n ({n})')

    dist_to_origin = np.sqrt((gene_embedding[:, :dims] ** 2).sum(axis=1))
    df = pd.DataFrame(gene_embedding[:, :dims], columns=[f'DM_{i + 1}' for i in range(dims)],
                      index=gene_names).assign(selected=other)
    n_genes = gene_embedding.shape[0]

    diffusion_mat = get_randow_walk_matrix(dist_mat, k=k)

    for i in range(n):
        if sum(df.selected == other) == 0:
            logger.warning(f"Early stop reached. {i - 1} gene trajectories were retrieved.")
            break

        dist_to_origin[df.selected != other] = -np.inf

        seed_idx = np.argmax(dist_to_origin)
        logger.info(f'Generating trajectory from {gene_names[seed_idx]}')

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
