# Python Gene Trajectory
This package is a Python implementation of [GeneTrajectory](https://github.com/KlugerLab/GeneTrajectory), 
which is implemented in R. 

Note that, although the implementation is equivalent, it will produce slightly different results to the R implementation
because the signs of eigenvectors may differ and because of the randomness of K-means during the `coarse_grain` step. 


# Install #
The development version of the package can be installed as 
```
pip install gene-trajectory
```

The development version of the package can be installed as 
```
pip install git+https://github.com/Klugerlab/GeneTrajectory-python.git
```

# Tutorials #
There are tutorials in Jupyter Notebook format in the
[notebooks](https://github.com/KlugerLab/GeneTrajectory-python/tree/main/notebooks) folder of the GitHub project. 
To get started, please follow the tutorial on Human myeloid cells on  
[tutorial_human_myeloid.ipynb](https://github.com/KlugerLab/GeneTrajectory-python/blob/main/notebooks/tutorial_human_myeloid.ipynb)

