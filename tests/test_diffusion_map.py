import unittest
import numpy as np
from sklearn.metrics import pairwise_distances

from gene_trajectory.diffusion_map import diffusion_map, get_symmetrized_affinity_matrix


class DiffusionMapTestCase(unittest.TestCase):
    def test_diffusion_map(self):
        """
        Small test

        ```
        ce  = t(matrix(c(0,0, 2,3, 3,2,  5,0, 0,5), nrow = 2,
                    dimnames = list(c("D1","D2"), c("C1", "C2", "C3", "C4", "C5"))))
        cgd = as.matrix(dist(ce, method = "manhattan"))

        > diffusion.map(cgd, K=3, nEV = 3)
        ```
        returns
        ```
        $diffu.emb
                EV.1          EV.2        EV.3
        C1 -0.292406  6.666994e-17 -0.15219742
        C2 -0.292406 -5.888965e-02  0.11845799
        C3 -0.292406  5.888965e-02  0.11845799
        C4 -0.292406  2.515596e-01 -0.07091361
        C5 -0.292406 -2.515596e-01 -0.07091361

        $eigen.vals
        [1] 1.0000000 0.5219069 0.3861848

        attr(,"class")
        [1] "diffusion.map" "list"
        ```
        """
        ce = np.array([[0, 0], [2, 3], [3, 2], [5, 0], [0, 5]])
        gd = pairwise_distances(ce, metric='manhattan')

        diffu_emb, eigen_vals = diffusion_map(gd, k=3, n_ev=3, t=1)

        self.assertEqual(3, len(eigen_vals))

        np.testing.assert_array_almost_equal([1.0000000, 0.5219069, 0.3861848], eigen_vals, 6)
        # The eigenvectors have opposite direction between Python and R
        np.testing.assert_array_almost_equal(
            np.abs([-0.292406, -0.292406, -0.292406, -0.292406, -0.292406]),
            np.abs(diffu_emb[:, 0]), 6)
        np.testing.assert_array_almost_equal(
            np.abs([6.666994e-17, -5.888965e-02, 5.888965e-02, 2.515596e-01, -2.515596e-01]),
            np.abs(diffu_emb[:, 1]), 6)
        np.testing.assert_array_almost_equal(
            np.abs([-0.15219742, 0.11845799, 0.11845799, -0.07091361, -0.07091361]),
            np.abs(diffu_emb[:, 2]), 6)

    @staticmethod
    def test_get_symmetrized_affinity_matrix():
        ce = np.array([[0, 0], [2, 3], [3, 2]])
        gd = pairwise_distances(ce, metric='manhattan')

        affinity = get_symmetrized_affinity_matrix(gd, k=3)
        np.testing.assert_almost_equal(
            np.abs([
                [1., 0.367879, 0.367879],
                [0.367879, 1., 0.852144],
                [0.367879, 0.852144, 1.]
            ]), affinity, 6)


if __name__ == '__main__':
    unittest.main()
