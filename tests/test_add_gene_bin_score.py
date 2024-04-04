import unittest

import numpy as np

from gene_trajectory.add_gene_bin_score import add_gene_bin_score
from tests.example_data import example_adata, diffusion_map, gene_trajectories, to_adata, gene_expression


class AddGeneScoreTestCase(unittest.TestCase):
    def test_add_gene_bin_score(self):

        adata = to_adata(gene_expression.T, var_names=list(gene_trajectories.index.values))

        add_gene_bin_score(adata, gene_trajectories, trajectories=1, n_bins=2)

        self.assertListEqual(['Trajectory1_genes1', 'Trajectory1_genes2'], adata.obs_keys())
        np.testing.assert_almost_equal([0.666667, 0.333333, 0.333333], adata.obs['Trajectory1_genes1'], decimal=5)
        np.testing.assert_almost_equal([0.5, 0.5, 0.5], adata.obs['Trajectory1_genes2'], decimal=5)


if __name__ == '__main__':
    unittest.main()
