# References
If you use this tool in your research and find it useful, you can cite the following reference 
from our paper {cite}`qu_gene_2024`.
```bibtex
@article{qu_gene_2024,
	title = {Gene trajectory inference for single-cell data by optimal transport metrics},
	issn = {1546-1696},
	url = {https://doi.org/10.1038/s41587-024-02186-3},
	doi = {10.1038/s41587-024-02186-3},
	abstract = {Single-cell RNA sequencing has been widely used to investigate cell state transitions and gene dynamics of biological processes. Current strategies to infer the sequential dynamics of genes in a process typically rely on constructing cell pseudotime through cell trajectory inference. However, the presence of concurrent gene processes in the same group of cells and technical noise can obscure the true progression of the processes studied. To address this challenge, we present GeneTrajectory, an approach that identifies trajectories of genes rather than trajectories of cells. Specifically, optimal transport distances are calculated between gene distributions across the cell–cell graph to extract gene programs and define their gene pseudotemporal order. Here we demonstrate that GeneTrajectory accurately extracts progressive gene dynamics in myeloid lineage maturation. Moreover, we show that GeneTrajectory deconvolves key gene programs underlying mouse skin hair follicle dermal condensate differentiation that could not be resolved by cell trajectory approaches. GeneTrajectory facilitates the discovery of gene programs that control the changes and activities of biological processes.},
	journal = {Nature Biotechnology},
	author = {Qu, Rihao and Cheng, Xiuyuan and Sefik, Esen and Stanley III, Jay S. and Landa, Boris and Strino, Francesco and Platt, Sarah and Garritano, James and Odell, Ian D. and Coifman, Ronald and Flavell, Richard A. and Myung, Peggy and Kluger, Yuval},
	month = apr,
	year = {2024},
}
```
The package makes use of the following packages
- [POT: Python Optimal Transport package](https://pythonot.github.io/) {cite}`flamary2021pot` for computing Earth Mover's distances
- [Scanpy](https://scanpy.readthedocs.io/en/stable/) for performing preprocessing and data storage {cite}`Wolf2018`


## Bibliography
```{bibliography}
```