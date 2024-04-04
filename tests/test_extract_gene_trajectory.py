import unittest

import numpy as np
import pandas as pd

from gene_trajectory.extract_gene_trajectory import get_gene_embedding, get_randow_walk_matrix, get_gene_pseudoorder, \
    extract_gene_trajectory
from tests.example_data import gene_names, gene_trajectories


class CoarseGrainTestCase(unittest.TestCase):
    gdm = np.array([
        [0.000000, 4.269544, 4.414329, 7.247308, 8.027305],
        [4.269544, 0.000000, 6.927429, 6.761171, 8.801408],
        [4.414329, 6.927429, 0.000000, 6.531824, 6.798761],
        [7.247308, 6.761171, 6.531824, 0.000000, 5.766003],
        [8.027305, 8.801408, 6.798761, 5.766003, 0.000000]])

    # Generated using R. Equal to get_gene_embedding() up to column signs
    gem = np.array([
        [0.17043479, 0.04309062, 0.02332959],
        [0.11273428, -0.10420329, 0.04087336],
        [0.02461065, 0.09818398, -0.06456570],
        [-0.12521597, -0.08515901, -0.07519815],
        [-0.20356317, 0.04675745, 0.08842471]])

    def test_get_gene_embedding(self):
        diffu_emb, eigen_vals = get_gene_embedding(self.gdm, k=3, n_ev=3)
        self.assertEqual(3, len(eigen_vals))

        np.testing.assert_array_almost_equal([0.4765176, 0.2777494, 0.2150586], eigen_vals, 6)

        # The eigenvectors have opposite direction between Python and R
        np.testing.assert_array_almost_equal(
            np.abs([0.170435, 0.112734, 0.024611, -0.125216, -0.203563]),
            np.abs(diffu_emb[:, 0]), 6)
        np.testing.assert_array_almost_equal(
            np.abs([0.043091, -0.104203, 0.098184, -0.085159, 0.046757]),
            np.abs(diffu_emb[:, 1]), 6)
        np.testing.assert_array_almost_equal(
            np.abs([0.023330, 0.040873, -0.064566, -0.075198, 0.088425]),
            np.abs(diffu_emb[:, 2]), 6)

    def test_get_random_walk_matrix(self):
        rw = np.array([
            [0.51517102, 0.18952083, 0.1832053, 0.06750703, 0.04459585],
            [0.22037146, 0.59903173, 0.0470527, 0.10012794, 0.03341618],
            [0.19758186, 0.04364105, 0.5555978, 0.10809274, 0.09508657],
            [0.07042714, 0.08983549, 0.1045631, 0.53745545, 0.19771881],
            [0.05148485, 0.03317748, 0.1017877, 0.21879730, 0.59475272],
        ])
        np.testing.assert_almost_equal(rw, get_randow_walk_matrix(self.gdm, k=2), 6)

    def test_get_gene_pseudoorder(self):
        np.testing.assert_array_equal([1, 2, 3, 4, 5], get_gene_pseudoorder(self.gdm, list(range(5)), 4))
        np.testing.assert_array_equal([5, 4, 3, 2, 1], get_gene_pseudoorder(self.gdm, list(range(5)), 0))
        np.testing.assert_array_equal([0, 3, 0, 2, 1], get_gene_pseudoorder(self.gdm, [1, 3, 4], 1))

    def test_extract_gene_trajectory(self):
        gt = extract_gene_trajectory(gene_embedding=self.gem, dist_mat=self.gdm, gene_names=gene_names,
                                     n=1, t_list=[3], dims=2)

        np.testing.assert_array_equal(gene_trajectories.index, gt.index)
        np.testing.assert_almost_equal(gene_trajectories['DM_1'], gt.DM_1.values, 6)
        np.testing.assert_almost_equal(gene_trajectories['DM_2'], gt.DM_2.values, 6)
        np.testing.assert_array_equal(gene_trajectories['selected'], gt.selected.values)
        np.testing.assert_array_equal(gene_trajectories['Pseudoorder-1'], gt['Pseudoorder-1'].values)


if __name__ == '__main__':
    unittest.main()
