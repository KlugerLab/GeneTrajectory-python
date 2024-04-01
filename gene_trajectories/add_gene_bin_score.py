from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc


def add_gene_bin_score(
        adata: sc.AnnData,
        gene_trajectory: pd.DataFrame,
        n_bins: int = 5,
        trajectories: int = 2,
        layer: Optional[str] = None,
        reverse: Optional[bool] = None,
        prefix: str = 'Trajectory',
) -> None:
    """
    Add gene bin score

    :param adata a scanpy Anndata object
    :param gene_trajectory Gene trajectory data frame
    :param n_bins How many gene bins
    :param trajectories: Which gene trajectories to define gene bin score
    :param layer string Which layer to use. Uses adata.X if empty
    :param reverse: Whether to reverse the order of genes along each trajectory
    :param prefix: String added to the names in the metadata
    """

    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f'Layer {layer} not found in adata. Available {list(adata.layers.keys())}')

    for trajectory in range(1, trajectories + 1):
        trajectory_name = f'Pseudoorder-{trajectory}'

        gene_trajectory_reordered = gene_trajectory.sort_values(trajectory_name)
        genes = gene_trajectory_reordered[gene_trajectory_reordered[trajectory_name] > 0].index.values
        if reverse:
            genes = list(reversed(genes))
        step = len(genes) / n_bins

        for i in range(n_bins):
            start = int(np.ceil(i * step))
            end = min(int(np.ceil((i + 1) * step)), len(genes))
            genes_subset = genes[start:end]

            adata_subset = adata[:, genes_subset]
            x = adata_subset.layers[layer] if layer else adata_subset.X
            normalized_gc = x > 0
            meta = np.squeeze(np.asarray(normalized_gc.sum(axis=1) / len(genes_subset)))
            adata.obs[f'{prefix}{trajectory}_genes{i + 1}'] = meta
