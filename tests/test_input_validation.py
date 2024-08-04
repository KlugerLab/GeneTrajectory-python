import unittest

import numpy as np

from gene_trajectory.util.input_validation import validate_matrix


class InputValidationTestCase(unittest.TestCase):
    def test_validate_matrix(self):
        m = np.array([[1, 2], [3, 4]])

        validate_matrix(m, min_value=1, max_value=4, square=True, shape=(2, 2))

        with self.assertRaisesRegex(ValueError, '.*does not have 3 rows.*'):
            validate_matrix(m, nrows=3)
        with self.assertRaisesRegex(ValueError, '.*does not have 8 columns.*'):
            validate_matrix(m, ncols=8)
        with self.assertRaisesRegex(ValueError, '.*does not have shape \\(1, 1\\)'):
            validate_matrix(m, shape=(1, 1))
        with self.assertRaisesRegex(ValueError, '.*Min_size: 3.*'):
            validate_matrix(m, min_size=3)
        with self.assertRaisesRegex(ValueError, '.*should not have values less than 5.*'):
            validate_matrix(m, min_value=5)
        with self.assertRaisesRegex(ValueError, '.*should not have values greater than 1.*'):
            validate_matrix(m, max_value=1)


if __name__ == '__main__':
    unittest.main()
