from typing import Iterable, Callable
from ipywidgets import VBox, HBox, Widget, IntSlider, interactive_output, Output, Label, Button, FloatSlider
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from gene_trajectory.extract_gene_trajectory import extract_gene_trajectory
from gene_trajectory.plot.gene_trajectory_plots import plot_gene_trajectory_3d


class ExtractGeneTrajectoryWidget(HBox):
    widgets: dict[str, Widget] = None
    t_list_widgets: list[IntSlider] = None
    dims_widget = IntSlider = None
    k_widget = IntSlider = None
    quantile_widget = FloatSlider = None
    f: Callable = None
    option_panel: VBox = None
    output: Output = None
    gene_trajectory: pd.DataFrame = None

    @property
    def dims(self) -> int:
        return self.dims_widget.value # noqa

    @property
    def k(self) -> int:
        return self.k_widget.value # noqa

    @property
    def quantile(self) -> float:
        return self.quantile_widget.value # noqa

    @property
    def tlist(self) -> list[int]:
        return [t.value for t in self.t_list_widgets] # noqa

    def __init__(self, gene_embedding: pd.DataFrame, dist_mat: np.array, gene_names: list[str],
                 t_list: Iterable[int] = (3, 3, 3),
                 dims=5,
                 k=10,
                 quantile=0.02):
        """
        An interactive widget for optimizing the parameters of extract_gene_trajectory

        :param gene_embedding: Gene embedding
        :param dist_mat: Gene-gene Wasserstein distance matrix (symmetric)
        :param gene_names:
        :param t_list:  Number of diffusion times to retrieve each trajectory
        :param n: Number of gene trajectories to retrieve. Will be set to the length of t_list
        :param dims: Dimensions of gene embedding to use to identify terminal genes (extrema)
        :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
        :param quantile: Thresholding parameter to extract genes for each trajectory. Default: 0.02
        :return: A widget
        """
        super().__init__()

        def build_plot(dims: int, k: int, quantile: float, **kwargs):
            tl = [v for n, v in kwargs.items() if n.startswith("_t_")]
            plt.figure(2, tight_layout=True, figsize=(6, 4))
            self.gene_trajectory = extract_gene_trajectory(gene_embedding, dist_mat, gene_names=gene_names,
                                                           t_list=tl, dims=dims, k=k, quantile=quantile)
            plot_gene_trajectory_3d(self.gene_trajectory)
            plt.show()

        self.f = build_plot

        self.build_widgets(t_list, dims, k, quantile)
        self.build_ui()

    def build_ui(self):
        def add_trajectory(_):
            self.build_widgets(self.tlist + [3], self.dims, self.k, self.quantiles)
            self.build_ui()

        def remove_trajectory(_):
            self.build_widgets(self.tlist[:-1], self.dims, self.k, self.quantiles)
            self.build_ui()

        add_btn = Button(text="", icon="plus")
        add_btn.on_click(add_trajectory)
        remove_btn = Button(text="", icon="minus")
        remove_btn.on_click(remove_trajectory)

        self.option_panel = VBox(children=[
            Label("Extract gene trajectories options"),
            self.dims_widget, self.k_widget, self.quantile_widget,
            Label("t_list"), *self.t_list_widgets, HBox([add_btn, remove_btn])])
        self.output = interactive_output(self.f, self.widgets)
        self.output.layout.height = '400px'

        self.children = (self.option_panel, self.output)

    def build_widgets(self, t_list: Iterable[int], dims: int, k: int, quantile: float):
        self.t_list_widgets = [IntSlider(t, 1, 10, description=f"Trajectory-{i + 1}")
                               for i, t in enumerate(t_list)]
        self.dims_widget = IntSlider(dims, 2, 20, description="dims")
        self.k_widget = IntSlider(k, 2, 10, description="k")
        self.quantile_widget = FloatSlider(value=quantile, min=0.001, max=0.1,  step=0.001, description="quantile")

        self.widgets = {"dims": self.dims_widget, "k": self.k_widget, "quantile": self.quantile_widget,
                        **{f"_t_{i}": t for i, t in enumerate(self.t_list_widgets)}}
