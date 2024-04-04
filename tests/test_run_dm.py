import unittest

import numpy as np

from gene_trajectory.run_dm import run_dm
from tests.example_data import example_adata, diffusion_map


class RunDmTestCase(unittest.TestCase):
    @staticmethod
    def test_run_dm():
        adata = example_adata()
        run_dm(adata)

        np.testing.assert_almost_equal(adata.obsm['X_dm'], diffusion_map, decimal=5)


if __name__ == '__main__':
    unittest.main()
