import unittest
import numpy as np
from helper.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()

    def test_calculate_3d_avg_euclid_error(self):
        truth_x, truth_y, truth_z = np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])
        est_x, est_y, est_z = np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])
        self.assertEqual(self.evaluator.calculate_3d_avg_euclid_error(truth_x, truth_y, truth_z, est_x, est_y, est_z),
                         0.0)

        est_x, est_y, est_z = np.array([2, 3, 4]), np.array([5, 6, 7]), np.array([8, 9, 10])
        expected_error = np.mean(np.sqrt(3))
        self.assertAlmostEqual(
            self.evaluator.calculate_3d_avg_euclid_error(truth_x, truth_y, truth_z, est_x, est_y, est_z),
            expected_error, places=6)

    def test_compute_average_stds(self):
        std_dict = {'x': [0.1, 0.2, 0.3], 'y': [0.2, 0.3, 0.4], 'z': [0.3, 0.4, 0.5]}
        overall_uncer, x_uncer, y_uncer, z_uncer = self.evaluator.compute_average_stds(std_dict)

        self.assertAlmostEqual(x_uncer, np.mean(std_dict['x']))
        self.assertAlmostEqual(y_uncer, np.mean(std_dict['y']))
        self.assertAlmostEqual(z_uncer, np.mean(std_dict['z']))
        self.assertAlmostEqual(overall_uncer, np.mean([x_uncer, y_uncer, z_uncer]))

    # sourced from scipy.org docs
    def test_average_mahalanobis_distance(self):
        test_cov = [[1.5, -0.5, -0.5], [-0.5, 1.5, -0.5], [-0.5, -0.5, 1.5]]
        pred_means = [np.array([0, 2, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])]
        pred_covs = [test_cov, test_cov, test_cov]
        truths = [np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([0, 1, 0])]

        expected_distance = (1 + 1 + 1.7320508) / 3
        self.assertAlmostEqual(self.evaluator.average_mahalanobis_distance(pred_means, pred_covs, truths),
                               expected_distance, places=6)

    def test_compute_average_covariance_trace(self):
        pred_covs = [
            np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]),
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        ]
        expected_trace = np.mean([np.trace(cov) for cov in pred_covs])
        self.assertAlmostEqual(self.evaluator.compute_average_covariance_trace(pred_covs), expected_trace, places=6)

    def test_compute_average_determinant(self):
        pred_covs = [
            np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]),
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        ]
        expected_determinant = np.mean([np.linalg.det(cov) for cov in pred_covs])
        self.assertAlmostEqual(self.evaluator.compute_average_determinant(pred_covs), expected_determinant, places=6)

if __name__ == '__main__':
    unittest.main()

