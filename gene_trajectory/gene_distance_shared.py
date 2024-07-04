import logging
from concurrent.futures import ThreadPoolExecutor
import os
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Sized

import numpy as np
import ot
from tqdm import tqdm

from gene_trajectory.util.input_validation import validate_matrix
from gene_trajectory.util.shared_array import SharedArray, PartialStarApply

logger = logging.getLogger()
_DEFAULT_NUMITERMAX = 50000

# Implementation notes:
# - The matrices can get relatively large so they are kept in memory as shared objects
# - Using ThreadPoolExecutor as ProcessPoolExecutor (which would make more sense) sometimes halts on Mac,
#   even when changing multiprocessing.set_start_method('fork')


def cal_ot_mat(
        ot_cost: np.array,
        gene_expr: np.array,
        gene_pairs: Optional[Sized] = None,
        num_iter_max=_DEFAULT_NUMITERMAX,
        show_progress_bar=True,
        processes: Optional[int] = None,
) -> np.array:
    """
    Calculate the earth mover distance matrix. Note that this step is computationally expensive
    and will be performed in parallel.

    :param ot_cost: the cost matrix
    :param gene_expr: the gene expression matrix
    :param gene_pairs: only compute the distance for the given pairs (0-indexed) (default: None).
           the distance entry for missing pairs will be set to 1000*max(computed_gene_distances)
    :param num_iter_max: the max number of iterations when computing the distance (see ot.emd2)
    :param show_progress_bar: shows a progress bar while running the computation (default: True)
    :param processes:the number of processes to use (defaults to the number of CPUs available)
    :return: the distance matrix
    """
    processes = int(processes) if isinstance(processes, float) else os.cpu_count()
    validate_matrix(gene_expr, obj_name='Gene Expression Matrix', min_value=0)
    ncells, ngenes = gene_expr.shape
    validate_matrix(ot_cost, obj_name='Cost Matrix', shape=(ncells, ncells), min_value=0)

    if show_progress_bar:
        logger.info(f'Computing emd distance..')

    if gene_pairs is None:
        pairs = ((i, j) for i in range(0, ngenes - 1) for j in range(i + 1, ngenes))
        npairs = (ngenes * (ngenes - 1)) // 2
    else:
        pairs = gene_pairs
        npairs = len(gene_pairs)
    emd_mat = np.full((ngenes, ngenes), fill_value=np.nan)

    with SharedMemoryManager() as manager:
        start_time = time.perf_counter()
        # create and configure the process pool

        with ThreadPoolExecutor(max_workers=processes) as pool:
            # prepare shared environment
            cost = SharedArray.copy(manager, np.asarray(ot_cost))
            gexp = SharedArray.copy(manager, np.asarray(gene_expr))
            f = PartialStarApply(_cal_ot, cost, gexp)

            # execute tasks and process results
            result_generator = pool.map(f,  ((num_iter_max, i, j) for i, j in pairs))
            if show_progress_bar:
                result_generator = tqdm(result_generator, total=npairs, position=0, leave=True)
            for d, i, j in result_generator:
                emd_mat[i, j] = emd_mat[j, i] = d
        finish_time = time.perf_counter()
        if show_progress_bar:
            logger.info("Program finished in {} seconds - using multiprocessing".format(finish_time - start_time))

        np.fill_diagonal(emd_mat, 0)
        np.nan_to_num(emd_mat, nan=1000 * np.nanmax(emd_mat), copy=False)

        return emd_mat


def _cal_ot(ot_cost_matrix: np.array, gene_expr_matrix: np.array, num_iter_max: int, i: int, j: int):
    """
    Computes the EMD between i and j
    """
    gene_i = np.array(gene_expr_matrix[:, i]).reshape(-1)
    gene_j = np.array(gene_expr_matrix[:, j]).reshape(-1)
    gene_i = gene_i / sum(gene_i)
    gene_j = gene_j / sum(gene_j)

    emd_dist = ot.emd2(gene_i[np.nonzero(gene_i)],
                       gene_j[np.nonzero(gene_j)],
                       ot_cost_matrix[np.nonzero(gene_i)[0], :][:, np.nonzero(gene_j)[0]],
                       numItermax=num_iter_max)
    # return the generated value
    return emd_dist, i, j
