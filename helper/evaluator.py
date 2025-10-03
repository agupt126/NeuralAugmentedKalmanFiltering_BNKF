import numpy as np
from scipy.spatial.distance import mahalanobis

class Evaluator:
    def __init__(self):
        pass

    def calculate_3d_avg_euclid_error(self, truth_x, truth_y, truth_z, est_x, est_y, est_z):
        errors = np.sqrt((truth_x - est_x) ** 2 + (truth_y - est_y) ** 2 + (truth_z - est_z) ** 2)
        errors = errors[errors <= 1000]  # filter out >1000
        if len(errors) > 0:
            return np.mean(errors)
        else:
            return np.nan

    def compute_average_stds(self, std_dict):
        x_uncer = np.mean(std_dict['x'])
        y_uncer = np.mean(std_dict['y'])
        z_uncer = np.mean(std_dict['z'])
        overall_uncer = np.mean([x_uncer, y_uncer, z_uncer])
        return overall_uncer, x_uncer, y_uncer, z_uncer

    def mahalanobis_chol(self, mean, truth, C, eps=1e-9):
        d = np.asarray(mean).reshape(-1) - np.asarray(truth).reshape(-1)
        C = 0.5 * (C + C.T)  # symmetrize
        C = C + eps * np.eye(C.shape[0])  # jitter to avoid singularity
        L = np.linalg.cholesky(C)  # will raise if still not SPD
        y = np.linalg.solve(L, d)
        return float(np.sqrt(y @ y))

    def average_mahalanobis_distance(self, pred_means, pred_covs, truths):
        total_distance = 0
        num_samples = 0

        for mean, cov, truth in zip(pred_means, pred_covs, truths):
            mean = np.array(mean).reshape(3, )
            truth = np.array(truth).reshape(3, )
            #cov_inv = np.linalg.inv(cov)
            try:
                distance = self.mahalanobis_chol(mean, truth, cov, eps=1e-9)
            except np.linalg.LinAlgError:
                continue  # skip truly bad covariances
            if not np.isnan(distance) and not distance > 10000:
                total_distance += distance;
                num_samples += 1

            # distance = mahalanobis(mean, truth, cov_inv)
            # if np.isnan(distance):
            #     continue

            # total_distance += distance
            # num_samples += 1
        return total_distance / num_samples if num_samples > 0 else np.nan



    def compute_average_covariance_trace(self, pred_covs):
        traces = [np.trace(cov) for cov in pred_covs]
        return np.mean(traces)

    def compute_average_determinant(self, pred_covs):
        total_det = 0
        num_samples = 0
        for cov in pred_covs:
            det = np.linalg.det(cov)
            if not np.isnan(det) and not det > 100000:
                total_det += det
                num_samples += 1

        return total_det/num_samples if num_samples > 0 else np.nan