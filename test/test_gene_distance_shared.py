import unittest

import numpy as np

from gene_trajectory.gene_distance_shared import cal_ot_mat


class GeneDistanceSharedTestCase(unittest.TestCase):
    gdm = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 2, 1]])

    # Generated using R. Equal to get_gene_embedding() up to column signs
    gem = np.array([
        [.3, .1, .6],
        [.2, .3, .5],
        [.6, .2, .2]])

    expected_emd = np.array([
        [0.0, 0.8, 1.0],
        [0.8, 0.0, 0.9],
        [1.0, 0.9, 0.0]])

    def test_gene_distance_shared(self):
        mt = cal_ot_mat(ot_cost=self.gdm, gene_expr=self.gem.T, show_progress_bar=False)
        np.testing.assert_almost_equal(self.expected_emd, mt, 6)


if __name__ == '__main__':
    unittest.main()
