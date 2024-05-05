from typing import Iterable, Callable
from ipywidgets import VBox, HBox, Widget, IntSlider, interactive_output, Output, Label, Button, FloatSlider, TagsInput, \
    Layout
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from gene_trajectory.extract_gene_trajectory import extract_gene_trajectory
from gene_trajectory.plot.gene_trajectory_plots import plot_gene_trajectory_3d


class ExtractGeneTrajectoryWidget(HBox):
    widgets: dict[str, Widget] = None
    t_list_widgets: list[IntSlider] = None
    dims_widget: IntSlider = None
    k_widget: IntSlider = None
    quantile_widget: FloatSlider = None
    label_genes: TagsInput = None

    f: Callable = None
    option_panel: VBox = None
    output: Output = None
    gene_trajectory: pd.DataFrame = None
    max_t: int = 20

    @property
    def dims(self) -> int:
        return self.dims_widget.value  # noqa

    @property
    def k(self) -> int:
        return self.k_widget.value  # noqa

    @property
    def quantile(self) -> float:
        return self.quantile_widget.value  # noqa

    @property
    def tlist(self) -> list[int]:
        return [t.value for t in self.t_list_widgets]  # noqa

    def __init__(
            self,
            gene_embedding: pd.DataFrame,
            dist_mat: np.array,
            gene_names: list[str],
            t_list: Iterable[int] = (3, 3, 3),
            label_genes=(),
            dims: int = 5,
            k: int = 10,
            quantile: float = 0.02,
            max_t: int = 20,
    ):
        """
        An interactive widget for optimizing the parameters of extract_gene_trajectory

        :param gene_embedding: Gene embedding
        :param dist_mat: Gene-gene Wasserstein distance matrix (symmetric)
        :param gene_names:
        :param t_list:  Number of diffusion times to retrieve each trajectory
        :param dims: Dimensions of gene embedding to use to identify terminal genes (extrema)
        :param label_genes: Gene labels to plot (default: ())
        :param k: Adaptive kernel bandwidth for each point set to be the distance to its `K`-th nearest neighbor
        :param quantile: Thresholding parameter to extract genes for each trajectory. Default: 0.02
        :param max_t: Maximum value for t sliders. Default: 20
        :return: A widget
        """
        super().__init__()
        self.max_t = max_t

        def build_plot(dims: int, k: int, quantile: float, label_genes: list[str], **kwargs):
            tl = [v for n, v in kwargs.items() if n.startswith("_t_")]
            plt.figure(2, tight_layout=True, figsize=(6, 4))
            self.gene_trajectory = extract_gene_trajectory(gene_embedding, dist_mat, gene_names=gene_names,
                                                           t_list=tl, dims=dims, k=k, quantile=quantile)
            plot_gene_trajectory_3d(self.gene_trajectory, label_genes=label_genes)
            plt.show()

        self.f = build_plot

        self.build_widgets(t_list, dims, k, quantile, label_genes, gene_names)
        self.build_ui()

    def build_ui(self):
        def add_trajectory(_):
            self.build_widgets(self.tlist + [3], self.dims, self.k, self.quantile)
            self.build_ui()

        def remove_trajectory(_):
            self.build_widgets(self.tlist[:-1], self.dims, self.k, self.quantile)
            self.build_ui()

        add_btn = Button(text="", icon="plus")
        add_btn.on_click(add_trajectory)
        remove_btn = Button(text="", icon="minus")
        remove_btn.on_click(remove_trajectory)

        self.option_panel = VBox(children=[
            Label("Extract gene trajectories options"),
            self.dims_widget, self.k_widget, self.quantile_widget,
            Label("t_list"), *self.t_list_widgets, HBox([add_btn, remove_btn]),
            Label("genes"), self.label_genes,
        ])

        self.output = interactive_output(self.f, self.widgets)
        self.output.layout.height = '400px'

        self.children = (self.option_panel, self.output)

    def build_widgets(self, t_list: Iterable[int], dims: int, k: int, quantile: float, label_genes: list[str],
                      gene_names: list[str]):
        self.t_list_widgets = [IntSlider(t, 1, self.max_t, description=f"Trajectory-{i + 1}")
                               for i, t in enumerate(t_list)]
        self.dims_widget = IntSlider(dims, 2, 20, description="dims")
        self.k_widget = IntSlider(k, 2, 10, description="k")
        self.quantile_widget = FloatSlider(value=quantile, min=0.001, max=0.1, step=0.001, description="quantile")
        self.label_genes = TagsInput(value=list(label_genes), allowed_tags=sorted(gene_names),
                                     allow_duplicates=False, layout=Layout(width='300px'))

        self.widgets = {"dims": self.dims_widget, "k": self.k_widget,
                        "quantile": self.quantile_widget, "label_genes": self.label_genes,
                        **{f"_t_{i}": t for i, t in enumerate(self.t_list_widgets)}}
