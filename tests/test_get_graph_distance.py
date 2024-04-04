import unittest

import numpy as np

from gene_trajectory.get_graph_distance import get_graph_distance
from tests.example_data import example_adata, diffusion_map, graph_distance


class CoarseGrainTestCase(unittest.TestCase):
    @staticmethod
    def test_get_graph_distance():
        adata = example_adata()
        adata.obsm['X_dm'] = diffusion_map

        gd = get_graph_distance(adata, k=3)
        np.testing.assert_almost_equal(gd, graph_distance, decimal=5)


if __name__ == '__main__':
    unittest.main()
