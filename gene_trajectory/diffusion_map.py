from typing import Union
import numpy as np

from gene_trajectory.util.input_validation import validate_matrix


def diffusion_map(
        dist_mat: np.array,
        k: int = 10,
        sigma: Union[float, np.array, list] = None,
        n_ev: int = 30,
        t: int = 1,
) -> tuple[np.array, np.array]:
    """
    Run a Diffusion Map

    :param dist_mat: Precomputed distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `K` is specified
    :param n_ev: Number of leading eigenvectors to export
    :param t: Number of diffusion times
    :return: the diffusion embedding and the eigenvalues
    """
    validate_matrix(dist_mat, square=True)

    affinity_matrix_symm = get_symmetrized_affinity_matrix(dist_mat=dist_mat, k=k, sigma=sigma)
    normalized_vec = np.sqrt(1 / affinity_matrix_symm.sum(axis=1))
    affinity_matrix_norm = (affinity_matrix_symm * normalized_vec * normalized_vec[:, None])

    n_ev = min(n_ev, affinity_matrix_norm.shape[0])
    eigs = np.linalg.eigh(affinity_matrix_norm, )

    dm = eigs.eigenvectors * normalized_vec[:, None] * eigs.eigenvalues ** t

    diffu_emb = dm[:, :-n_ev - 1:-1]
    eigen_vals = eigs.eigenvalues[:-n_ev - 1:-1]

    return diffu_emb, eigen_vals

# Notes:
# scipy.linalg.eigh(affinity_matrix_norm, subset_by_index=[n - n_ev, n - 1]) could also be used but is 2x slower


def get_symmetrized_affinity_matrix(
        dist_mat: np.array,
        k: int = 10,
        sigma: Union[float, np.array, list] = None,
):
    """
    Computes the symmetrized distance matrix
    :param dist_mat: Precomputed distance matrix (symmetric)
    :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
    :param sigma: Fixed kernel bandwidth, `sigma` will be ignored if `k` is specified

    :return:
    """
    validate_matrix(dist_mat, square=True)

    dists = np.nan_to_num(dist_mat, 1e-6) # noqa
    k = min(k, dist_mat.shape[0])

    if sigma is None:
        sigma = np.apply_along_axis(func1d=sorted, axis=1, arr=dists)[:, k - 1]  # noqa
    elif np.isscalar(sigma):
        sigma = np.full(dists.shape[0], sigma)

    affinity_matrix = np.exp(-dists ** 2 / (sigma ** 2)[:, None])
    affinity_matrix_symm = (affinity_matrix + affinity_matrix.T) / 2
    return affinity_matrix_symm
