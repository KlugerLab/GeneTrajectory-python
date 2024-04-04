import os
import unittest
from tempfile import TemporaryDirectory
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from gene_trajectory.compute_gene_distance_cmd import cal_ot_mat


class DiffusionMapTestCase(unittest.TestCase):
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

    def test_compute_gene_distance_cmd(self):
        with TemporaryDirectory() as d:
            np.savetxt(os.path.join(d, 'ot_cost.csv'), self.gdm, delimiter=',')
            sio.mmwrite(os.path.join(d, 'gene_expression.mtx'), coo_matrix(self.gem.T))

            cal_ot_mat(d, show_progress_bar=False)

            res = np.loadtxt(d + "/emd.csv", delimiter=",")
            np.testing.assert_almost_equal(res, self.expected_emd, decimal=6)


if __name__ == '__main__':
    unittest.main()
