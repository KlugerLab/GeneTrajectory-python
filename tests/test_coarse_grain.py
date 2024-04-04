import unittest

import numpy as np
from sklearn.metrics import pairwise_distances

from gene_trajectory.coarse_grain import coarse_grain, select_top_genes
from tests.example_data import example_adata, random_adata


class CoarseGrainTestCase(unittest.TestCase):
    @staticmethod
    def test_cg():
        """
        Generated in R using
        https://github.com/KlugerLab/GeneTrajectory/blob/main/R/coarse.grain.R

        ```R
        random.seed = 1
        ce  = t(matrix(c(0,0, 2,3, 3,2,  5,0, 0,5), nrow = 2,
                dimnames = list(c("C1", "C2", "C3", "C4", "C5"), c("D1","D2"))))
        ge  = matrix(c(2,0,3,0,1, 0,3,0,0,1,  1,0,0,0,1), ncol = 3,
                     dimnames = list(c("C1", "C2", "C3", "C4", "C5"), c("G1","G2","G3")))
        cgd = as.matrix(dist(ce, method = "manhattan"))

        km.res <- stats::kmeans(ce, 4)
        km.res$cluster

        KNN.membership.mat <- matrix(0, nrow = 4, ncol = nrow(ce))
        for (i in 1:ncol(KNN.membership.mat)){
          KNN.membership.mat[km.res$cluster[i], i] <- 1
        }
        KNN.membership.mat <- KNN.membership.mat/apply(KNN.membership.mat, 1, sum)

        geu <- biclust::binarize(KNN.membership.mat, 0) %*% ge
        gdu <- KNN.membership.mat %*% cgd %*% t(KNN.membership.mat)
        ft  <- colnames(geu)
        ```
        """
        ce = np.array([[0, 0], [2, 3], [3, 2], [5, 0], [0, 5]])
        ge = np.array([[2, 0, 1], [0, 3, 0], [3, 0, 0], [0, 0, 0], [1, 1, 1]])
        n = 4

        gd = pairwise_distances(ce, metric='manhattan')

        cluster = np.array([2, 3, 3, 0, 1])
        gene_expression_updated, graph_dist_updated = coarse_grain(cell_embedding=ce,
                                                                   gene_expression=ge,
                                                                   graph_dist=gd,
                                                                   n=n,
                                                                   cluster=cluster)

        np.testing.assert_array_equal(np.array([[0, 0, 0], [1, 1, 1], [2, 0, 1], [3, 3, 0]]), gene_expression_updated)
        np.testing.assert_array_equal(np.array([[0, 10, 5, 5], [10, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 1]]),
                                      graph_dist_updated)

    @staticmethod
    def test_cg2():
        """
        Same as test_cg but with a different k-mean result
        """
        ce = np.array([[0, 0], [2, 3], [3, 2], [5, 0], [0, 5]])
        ge = np.array([[2, 0, 1], [0, 3, 0], [3, 0, 0], [0, 0, 0], [1, 1, 1]])
        n = 4

        gd = pairwise_distances(ce, metric='manhattan')

        gene_expression_updated, graph_dist_updated = coarse_grain(cell_embedding=ce,
                                                                   gene_expression=ge,
                                                                   graph_dist=gd,
                                                                   n=n,
                                                                   random_seed=1)

        np.testing.assert_array_equal(np.array([[3, 3, 0], [1, 1, 1], [2, 0, 1], [0, 0, 0]]), gene_expression_updated)
        np.testing.assert_array_equal(np.array([[1, 5, 5, 5], [5, 0, 5, 10], [5, 5, 0, 5], [5, 10, 5, 0]]),
                                      graph_dist_updated)

    def test_select_top_genes(self):
        adata = random_adata(shape=(100, 3000), seed=123)

        genes = select_top_genes(adata, layer='counts')

        self.assertListEqual(['Gene_204', 'Gene_971', 'Gene_1915', 'Gene_1775'], genes.tolist())


if __name__ == '__main__':
    unittest.main()
