# Python Gene Trajectory
This package is a Python implementation of [GeneTrajectory](https://github.com/KlugerLab/GeneTrajectory), 
which is implemented in R. 

Note that, although the implementation is equivalent, it will produce slightly different results to the R implementation
because the signs of eigenvectors may differ and because of the randomness of K-means during the `coarse_grain` step. 


# Install #
The package can be installed as 
```
pip install git+https://github.com/Klugerlab/PGT.git
```

# Tutorial # 
Please follow the Jupyter Notebook tutorial in [tutorial_e14.ipynb](notebooks/tutorial_e14.ipynb).
