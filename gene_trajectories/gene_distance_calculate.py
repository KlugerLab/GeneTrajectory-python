import os
import ot
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

_ot_cost_matrix: np.array = None
_gene_expr_matrix: np.matrix = None
_DEFAULT_NUMITERMAX = 50000
_num_iter_max = _DEFAULT_NUMITERMAX


def _cal_ot(ij):
    i, j = ij
    # compute the distance
    gene_i = np.array(_gene_expr_matrix[:, i]).reshape(-1)
    gene_j = np.array(_gene_expr_matrix[:, j]).reshape(-1)
    gene_i = gene_i / sum(gene_i)
    gene_j = gene_j / sum(gene_j)

    emd_dist = ot.emd2(gene_i[np.nonzero(gene_i)],
                       gene_j[np.nonzero(gene_j)],
                       np.array(_ot_cost_matrix)[np.nonzero(gene_i)[0], :][:, np.nonzero(gene_j)[0]],
                       numItermax=_num_iter_max)
    # return the generated value
    return emd_dist


def init_global_from_path(path, num_iter_max=_DEFAULT_NUMITERMAX):
    global _ot_cost_matrix
    global _gene_expr_matrix
    global _num_iter_max
    _ot_cost_matrix = np.loadtxt(open(path + "/ot_cost.csv", "rb"), delimiter=",")
    _gene_expr_matrix = sio.mmread(path + "/gene_expression.mtx").todense()
    num_iter_max = int(num_iter_max) if isinstance(num_iter_max, float) else _DEFAULT_NUMITERMAX
    _num_iter_max = num_iter_max


def save_result_to_path(emd_mat2, path):
    np.savetxt(path + "/emd.csv", emd_mat2, delimiter=",")


def init_global_from_np(ot_cost: np.array, gene_expr: np.array, num_iter_max=_DEFAULT_NUMITERMAX):
    global _ot_cost_matrix
    global _gene_expr_matrix
    global _num_iter_max
    _ot_cost_matrix = ot_cost
    _gene_expr_matrix = gene_expr
    num_iter_max = int(num_iter_max) if isinstance(num_iter_max, float) else _DEFAULT_NUMITERMAX
    _num_iter_max = num_iter_max


def cal_ot_mat(show_progress_bar=True,
               processes: int = None):
    processes = int(processes) if isinstance(processes, float) else os.cpu_count()
    print(f'Computing emd distance..')

    n = _gene_expr_matrix.shape[1]
    npairs = (n * (n - 1)) // 2

    start_time = time.perf_counter()
    # create and configure the process pool
    with ThreadPoolExecutor(max_workers=processes) as pool:
        # prepare arguments
        items = ((i, j) for i in range(0, n - 1) for j in range(i + 1, n))
        # execute tasks and process results in order
        # result = pool.starmap(cal_ot, tqdm(items))
        result_generator = pool.map(_cal_ot, items)
        if show_progress_bar:
            result_generator = tqdm(result_generator, total=npairs, position=0, leave=True)
        result = list(result_generator)
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time - start_time))
    print("---")

    ind = 0
    emd_mat = np.zeros((n, n))
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            emd_mat[i, j] = result[ind]
            ind += 1

    emd_mat2 = emd_mat + np.transpose(emd_mat)
    return emd_mat2
