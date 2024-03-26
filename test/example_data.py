import numpy as np
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

gene_expression = np.array([
    [2, 0, 1],
    [0, 3, 0],
    [3, 0, 0],
    [0, 0, 0],
    [1, 1, 1]
])

diffusion_map = np.array([
    [-0.04764353, 0.00940602, -0.00042641, -0.00158765],
    [0.11828658, 0.04134494, -0.00401907, -0.00012575],
    [-0.06615087, 0.03891922, -0.00681983,  0.00119593],
    [0.01417467, -0.05808308, -0.01944058,  0.0002601],
    [0.00654969, -0.02393814,  0.02780562,  0.00040004]
])

graph_distance = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 2, 1, 1],
    [1, 2, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
])



def example_adata() -> sc.AnnData:
    return to_adata(gene_expression)


def to_adata(x: np.array):
    adata = sc.AnnData(csr_matrix(np.asarray(x, dtype=np.float32)))
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata
