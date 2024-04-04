import argparse
import os
import scipy.io as sio
from scipy.sparse import issparse
import numpy as np

from gene_trajectory.gene_distance_shared import _DEFAULT_NUMITERMAX
from gene_trajectory.gene_distance_shared import cal_ot_mat as cal_ot_mat_from_numpy


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate the earth mover distance matrix')
    parser.add_argument('--path', type=os.path.abspath, required=True,
                        help='The path containing the cost matrix (ot_cost.csv)' +
                             ' and the gene expression matrix (gene_expression.mtx)')
    parser.add_argument('--num_iter_max', type=int, default=_DEFAULT_NUMITERMAX,
                        help='The max number of iterations when computing the distance (see ot.emd2)')
    parser.add_argument('--show_progress_bar', type=bool, default=True,
                        help='Shows a progress bar while running the computation (default: True)')
    parser.add_argument('--processes', type=int, required=False, default=None,
                        help='The number of processes to use (defaults to the number of CPUs available)')
    return parser.parse_args()


def cal_ot_mat(
    path: str,
    num_iter_max=_DEFAULT_NUMITERMAX,
    show_progress_bar=True,
    processes: int = None,
) -> None:
    """
    Calculate the earth mover distance matrix. Note that this step is computationally expensive
    and will be performed in parallel.

    :param path: path to the folder where the cost matrix (ot_cost.csv) and the gene expression matrix
                 (gene_expression.mtx) are saved
    :param num_iter_max: the max number of iterations when computing the distance (see ot.emd2)
    :param show_progress_bar: shows a progress bar while running the computation (default: True)
    :param processes:the number of processes to use (defaults to the number of CPUs available)
    """
    ot_cost = np.loadtxt(os.path.join(path, "ot_cost.csv"), delimiter=",")
    gene_expr = sio.mmread(os.path.join(path,"gene_expression.mtx"))
    if issparse(gene_expr):
        gene_expr = gene_expr.todense()
    gene_pairs_file = os.path.join(path, "gene_pairs.csv")
    gene_pairs = np.loadtxt(gene_pairs_file, delimiter=",").astype(int) if os.path.isfile(gene_pairs_file) else None

    emd_mat2 = cal_ot_mat_from_numpy(ot_cost=ot_cost, gene_expr=gene_expr,
                                     gene_pairs=gene_pairs, num_iter_max=num_iter_max,
                                     show_progress_bar=show_progress_bar, processes=processes)
    np.savetxt(path + "/emd.csv", emd_mat2, delimiter=",")


if __name__ == '__main__':
    args = parse_args()

    cal_ot_mat(path=args.path,
               num_iter_max=args.num_iter_max,
               show_progress_bar=args.show_progress_bar,
               processes=args.processes)
