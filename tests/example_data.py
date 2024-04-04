import numpy as np
import pandas as pd
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

gene_names = ["Grin2a", "Sox2", "Cxcr4", "Cdkn1a", "Plk2"]

gene_trajectories = pd.DataFrame({
    'DM_1': [0.170435, 0.112734, 0.024611, -0.125216, -0.203563],
    'DM_2': [0.043091, -0.104203, 0.098184, -0.085159, 0.046757],
    'selected': ['Trajectory-1']*5,
    'Pseudoorder-1': [1, 2, 3, 4, 5],
}, index=gene_names)


def example_adata() -> sc.AnnData:
    return to_adata(gene_expression)


def to_adata(x: np.array, obs_names: list[str] = None, var_names: list[str] = None):
    adata = sc.AnnData(csr_matrix(np.asarray(x, dtype=np.float32)))
    adata.obs_names = obs_names or [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = var_names or [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata


def random_adata(shape=(100, 2000), seed=123) -> sc.AnnData:
    """
    A reasonably sized Scanpy element
    https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html
    """
    prng = np.random.RandomState(seed)
    counts = csr_matrix(prng.poisson(1, size=shape), dtype=np.float32)

    adata = sc.AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

    adata.raw = adata
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.raw.X.copy()

    return adata
