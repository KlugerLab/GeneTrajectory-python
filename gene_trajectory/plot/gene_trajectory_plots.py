import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Iterable, Any


def plot_gene_trajectory_3d(
        gene_trajectory: pd.DataFrame,
        s: int = 1,
        label_genes: Iterable[str] = None,
        **kwargs: Any
) -> None:
    """
    Generate a 3D plot of a gene-trajectory object

    :param gene_trajectory: a gene trajectory result
    :param s: scatterplot point size (default: 1)
    :param label_genes: Gene labels to plot (default: None)
    :param kwargs: plot arguments that will be passed to Axes.scatter
    """
    for c in 'DM_1', 'DM_2', 'DM_3', 'selected':
        if c not in gene_trajectory.columns:
            raise ValueError(f'Column {c} is not present in gene trajectory DataFrame')

    ax = plt.axes(projection='3d')
    selections = pd.Categorical(gene_trajectory.selected)

    for c in selections.categories:
        idxs = selections == c
        ax.scatter(xs=gene_trajectory['DM_1'][idxs],
                   ys=gene_trajectory['DM_2'][idxs],
                   zs=gene_trajectory['DM_3'][idxs],
                   s=s,
                   label=c,
                   **kwargs)

    if label_genes:
        for g in label_genes:
            ax.text(x=gene_trajectory['DM_1'][g],
                    y=gene_trajectory['DM_2'][g],
                    z=gene_trajectory['DM_3'][g],
                    s=g)
    ax.legend()


def plot_gene_trajectory_2d(
        gene_trajectory: pd.DataFrame,
        s: int = 1,
        label_genes: Iterable[str] = None,
        **kwargs: Any
) -> None:
    """
    Generate a 2D plot of a gene-trajectory object

    :param gene_trajectory: a gene trajectory result
    :param s: scatterplot point size (default: 1)
    :param label_genes: Gene labels to plot (default: None)
    :param kwargs: plot arguments that will be passed to sns.scatterplot
    """
    for c in 'DM_1', 'DM_2', 'selected':
        if c not in gene_trajectory.columns:
            raise ValueError(f'Column {c} is not present in gene trajectory DataFrame')
    sns.scatterplot(data=gene_trajectory,
                    x='DM_1',
                    y='DM_2',
                    hue='selected',
                    s=s,
                    **kwargs)
    if label_genes:
        ax = plt.gca()
        for g in label_genes:
            ax.text(x=gene_trajectory['DM_1'][g],
                    y=gene_trajectory['DM_2'][g],
                    s=g)


def plot_gene_trajectory_umap(
        adata: sc.AnnData,
        trajectory: str = 'Trajectory1',
        other_panels: Iterable[str] = (),
        reverse: bool = False,
        cmap: str = 'RdYlBu_r',
        **kwargs: Any,
) -> None:
    """
    Generate a series of umap plot for gene trajectory bins

    :param adata: a dataset with gene trajectory metadata
    :param trajectory: the name of the trajectory
    :param other_panels: other panels to be added to the umap
    :param reverse: reverse the order of the panels
    :param cmap: colormap to be used
    :param kwargs: plot arguments that will be passed to scanpy.pl.umap
    """
    other_panels = [other_panels] if isinstance(other_panels, str) else other_panels
    trajectory_panels = [k for k in adata.obs_keys() if k.startswith(trajectory)]
    if not trajectory_panels:
        raise ValueError(f'No obj metadata found for {trajectory}')
    if reverse:
        trajectory_panels.reverse()
    panels = [*trajectory_panels, *other_panels]
    sc.pl.umap(adata, color=panels, cmap=cmap, **kwargs)
