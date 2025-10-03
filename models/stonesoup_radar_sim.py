from collections import defaultdict

import numpy as np
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.detection import Detection
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.models.measurement.nonlinear import RangeRangeRateBinning

class StonesoupRadarSim:
    def __init__(self, proc_noise=0.1, sensor_noise_level='low', include_range_rate=False):
        self.proc_noise = proc_noise
        self.bearing_std, self.elev_std, self.range_std, self.range_rate_std = self.__return_noise_params(sensor_noise_level)
        self.radar_measurements = []

        # elevation, bearing, range, range_rate [radians, radians, m, m/s]
        if include_range_rate:
            self.measurement_model = RangeRangeRateBinning(
                range_res=self.range_std,
                range_rate_res=self.range_rate_std,
                ndim_state=6,
                mapping=(0, 2, 4),
                velocity_mapping=(1, 3, 5),
                noise_covar=np.diag([self.elev_std ** 2, self.bearing_std ** 2, self.range_std ** 2, self.range_rate_std ** 2]),
                translation_offset=np.array([[0], [0], [0]])
            )
        else:
            self.measurement_model = CartesianToElevationBearingRange(
                ndim_state=6,
                mapping=(0, 2, 4),
                noise_covar=np.diag([self.elev_std ** 2, self.bearing_std ** 2, self.range_std ** 2]),
                translation_offset=np.array([[0], [0], [0]])
            )

        self.incl_velocity = include_range_rate

    def __return_noise_params(self, a_level):
        match a_level.lower():
            case "low":
                bearing_std, elev_std, range_std, range_rate_std = 0.001, 0.001, 1, 0.01
            case "mid":
                bearing_std, elev_std, range_std, range_rate_std = 0.01, 0.01, 10, 0.1
            case "high":
                bearing_std, elev_std, range_std, range_rate_std = 0.1, 0.1, 100, 1
        return bearing_std, elev_std, range_std, range_rate_std

    def simulate_radar_measurements(self, trajectory, sensor_x=0, sensor_y=0, sensor_z=0):
        # MEASUREMENTS
        radar_measurements = []
        for index, state in enumerate(trajectory.truth_state_traj):
            radar_measurements.append(
                Detection(self.measurement_model.function(state, noise=True), timestamp=state.timestamp,
                          measurement_model=self.measurement_model))
        return radar_measurements

    def run_ekf_filter(self, trajectory, radar_measurements):
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42),
             ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42),
             ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42)])  # Process Noise Input
        ekf_predictor = ExtendedKalmanPredictor(transition_model)
        ekf_updater = ExtendedKalmanUpdater(self.measurement_model)

        generated_track_states = Track()
        var_pos = self.range_std**2
        var_vel = self.range_rate_std**2
        prior = GaussianState(
            [[trajectory.truth_x[0]], [trajectory.truth_vx[0]], [trajectory.truth_y[0]],
             [trajectory.truth_vy[0]], [trajectory.truth_z[0]], [trajectory.truth_vz[0]]],
            np.diag([var_pos, var_vel, var_pos, var_vel, var_pos, var_vel]), timestamp=trajectory.start_time)

        for index, measurement in enumerate(radar_measurements):
            if index == 0:  # no measurement at 0th step
                continue
            prediction = ekf_predictor.predict(prior, timestamp=measurement.timestamp)
            if not trajectory.measurement_only_eval and index in trajectory.truth_time_data_filtered_indices:
                generated_track_states.append(prediction)
                continue
            hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
            post = ekf_updater.update(hypothesis)
            generated_track_states.append(post)
            prior = generated_track_states[-1]
        return generated_track_states


    # def run_ukf_filter(self, trajectory, radar_measurements):
    #     transition_model = CombinedLinearGaussianTransitionModel(
    #         [ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42),
    #          ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42),
    #          ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42)])  # Process Noise Input
    #
    #     predictor = UnscentedKalmanPredictor(transition_model)
    #     updater = UnscentedKalmanUpdater(self.measurement_model)
    #
    #     generated_track_states = Track()
    #     var_pos = self.range_std**2
    #     var_vel = self.range_rate_std**2
    #     prior = GaussianState(
    #         [[trajectory.truth_x[0]], [trajectory.truth_vx[0]], [trajectory.truth_y[0]],
    #          [trajectory.truth_vy[0]], [trajectory.truth_z[0]], [trajectory.truth_vz[0]]],
    #         np.diag([var_pos, var_vel, var_pos, var_vel, var_pos, var_vel]), timestamp=trajectory.start_time)
    #
    #     for index, measurement in enumerate(radar_measurements):
    #         if index == 0:  # no measurement at 0th step
    #             continue
    #         prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    #         if not trajectory.measurement_only_eval and index in trajectory.truth_time_data_filtered_indices:
    #             generated_track_states.append(prediction)
    #             continue
    #         hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    #         post = updater.update(hypothesis)
    #         generated_track_states.append(post)
    #         prior = generated_track_states[-1]
    #     return generated_track_states

    def run_ukf_filter(self, trajectory, radar_measurements):
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(noise_diff_coeff=self.proc_noise, seed=42)] * 3
        )

        predictor = UnscentedKalmanPredictor(
            transition_model,
            alpha=1e-3,  # pull sigma points closer
            beta=2.0,
            kappa=0.0
        )

        updater = UnscentedKalmanUpdater(
            self.measurement_model,
            alpha=1e-3,
            beta=2.0,
            kappa=0.0
        )

        def _pdize(P, eps=1e-12):
            P = 0.5 * (P + P.T)
            np.fill_diagonal(P, np.maximum(np.diag(P), eps))
            return P + eps * np.eye(P.shape[0])

        pos_std0 = max(self.range_std, 1.0)
        vel_std0 = max(5.0 * abs(trajectory.truth_vx[0]) + 1.0, 5.0)
        P0 = np.diag([pos_std0 ** 2, vel_std0 ** 2,
                      pos_std0 ** 2, vel_std0 ** 2,
                      pos_std0 ** 2, vel_std0 ** 2]).astype(float)
        P0 = _pdize(P0)

        prior = GaussianState(
            np.array([[trajectory.truth_x[0]], [trajectory.truth_vx[0]],
                      [trajectory.truth_y[0]], [trajectory.truth_vy[0]],
                      [trajectory.truth_z[0]], [trajectory.truth_vz[0]]], dtype=float),
            P0, timestamp=trajectory.start_time
        )

        generated_track_states = Track()
        for index, measurement in enumerate(radar_measurements):
            if index == 0:
                continue

            prediction = predictor.predict(prior, timestamp=measurement.timestamp)
            prediction = prediction.__class__(prediction.state_vector, _pdize(prediction.covar), prediction.timestamp)

            if (not trajectory.measurement_only_eval) and (index in trajectory.truth_time_data_filtered_indices):
                generated_track_states.append(prediction)
                prior = prediction
                continue

            # Update with soft recovery
            try:
                post = updater.update(SingleHypothesis(prediction, measurement))
            except np.linalg.LinAlgError:
                prediction.covar = _pdize(prediction.covar * 10.0)  # inflate & fix PD
                post = updater.update(SingleHypothesis(prediction, measurement))

            post.covar = _pdize(post.covar)  # <-- in-place
            generated_track_states.append(post)
            prior = post

        return generated_track_states

    def convert_radar_observations_to_cartesian(self, observations):
        if self.incl_velocity:
            substates = ('x', 'vx', 'y', 'vy', 'z', 'vz')
            mapping = (0, 1, 2, 3, 4, 5)
        else:
            substates = ('x', 'y', 'z')
            mapping = (0, 2, 4)

        converted_state_vecs = []
        dict_breakdown = defaultdict(list)
        for i_index, detection in enumerate(observations):
            meas_model = detection.measurement_model
            converted_state_vecs.append(np.array(meas_model.inverse_function(detection)[mapping, :]))
            for j_index, value in enumerate(mapping):
                dict_breakdown[substates[j_index]].append(converted_state_vecs[i_index][j_index])
        return converted_state_vecs, dict_breakdown

    def process_stonesoup_tracker_info(self, a_track_list):
        predicted_states = defaultdict(list)
        predicted_uncer_std = defaultdict(list)

        for a_track in a_track_list:
            predicted_states['x'].append(a_track.state_vector[0])
            predicted_states['y'].append(a_track.state_vector[2])
            predicted_states['z'].append(a_track.state_vector[4])

            predicted_uncer_std['x'].append(np.sqrt(a_track.covar[0, 0]))
            predicted_uncer_std['y'].append(np.sqrt(a_track.covar[2, 2]))
            predicted_uncer_std['z'].append(np.sqrt(a_track.covar[4, 4]))
        return predicted_states, predicted_uncer_std

    def gather_position_prediction_info(self, a_track_list):
        full_pos_states = []
        full_pos_covs = []
        pos_indices = (0, 2, 4)
        for a_track in a_track_list:
            full_pos_states.append((a_track.state_vector[0], a_track.state_vector[2], a_track.state_vector[4]))
            full_pos_covs.append(a_track.covar[np.ix_(pos_indices, pos_indices)])
        return full_pos_states, full_pos_covs
