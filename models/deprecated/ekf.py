import numpy as np
from tqdm import tqdm
from collections import defaultdict


class EKF:
    """

        Author: Addison Sears-Collins
        https://automaticaddison.com
        Description: Extended Kalman Filter

    """

    def __init__(self, state_size=3, sensor_measurement_uncertainty=0.01, state_space_process_noise=1e-6,
                 update_space_process_noise=1e-6, show_info=False):

        ## GENERIC ##
        np.set_printoptions(precision=3, suppress=True)  # Supress scientific notation when printing NumPy arrays
        self.state_size = state_size
        self.show_info = show_info

        ## STATE SPACE MODEL PARAMETERS ##
        self.A_k_minus_1 = np.identity(
            state_size)  # Expresses how the state of the system [x,y,z] changes from k-1 to k when no control command is executed.
        self.process_noise_v_k_minus_1 = np.full((state_size,), state_space_process_noise)

        ## OBSERVATION MODEL PARAMETERS ##
        self.Q_k = update_space_process_noise * np.identity(state_size)  # State model process noise
        self.H_k = np.identity(state_size)  # measurement matrix
        self.R_k = (sensor_measurement_uncertainty ** 2) * np.identity(state_size)  # Sensor uncertainty [sigma^2]
        self.sensor_noise_w_k = np.full((state_size,), 0)  # no sensor bias

    def forward(self, z_k_observation_vector, state_estimate_k_minus_1, control_vector_k_minus_1, delta_t,
                p_k_minus_1=None):
        """
        Extended Kalman Filter. Fuses noisy sensor measurement to
        create an optimal estimate of the state of the robotic system.

        INPUT
            :param z_k_observation_vector The observation/noisy measurement
                3x1 NumPy Array [x,y,z] in the global reference frame
                in [m,m,m].

            :param state_estimate_k_minus_1 The state estimate at time k-1
                3x1 NumPy Array [x,y,z] in the global reference frame
                in [m,m,m].

            :param control_vector_k_minus_1 The control vector applied at time k-1
                3x1 NumPy Array [vx,vy,vz] in the global reference frame
                in [m/s, m/s, m/s].

            :param p_k_minus_1 The state covariance matrix estimate at time k-1
                3x3 NumPy Array [zero correlation]

            :param delta_t Time interval in seconds

        OUTPUT
            :return state_estimate_k near-optimal state estimate at time k
                3x1 NumPy Array ---> [m,m,m]
            :return P_k state covariance_estimate for time k
                3x3 NumPy Array
        """
        ######################### Predict #############################
        state_space_estimate = self.state_space_model(state_estimate_k_minus_1, control_vector_k_minus_1, delta_t)

        ################### Update (Correct) ##########################
        if not isinstance(p_k_minus_1, np.ndarray):
            p_k_minus_1 = 0.001 * np.identity(self.state_size)  # start with 1 meter uncertainty

        ekf_state_estimate, cov_est = self.observation_model(state_space_estimate, z_k_observation_vector, p_k_minus_1)

        if self.show_info:
            print(f'State Estimate Before EKF={state_space_estimate}')
            print(f'Observation={z_k_observation_vector}')
            print(f'State Estimate After EKF={ekf_state_estimate}')

        return ekf_state_estimate, cov_est

    def getB(self, delta_t):
        """
        3x3 matix -> number of states x number of control inputs

        Expresses how the state of the system [x,y,z] changes
        from k-1 to k due to the control commands (i.e. velocity).

        :param deltak: The change in time from time step k-1 to k in sec
        """
        B = delta_t * np.identity(
            self.state_size)
        return B

    def state_space_model(self, state_estimate_k_minus_1, control_vector_k_minus_1, delta_t):

        # Predict the state estimate at time k based on the state
        # estimate at time k-1 and the control input applied at time k-1.
        state_space_estimate = (self.A_k_minus_1 @ state_estimate_k_minus_1 +
                                (self.getB(delta_t)) @ control_vector_k_minus_1 +
                                self.process_noise_v_k_minus_1)
        return state_space_estimate

    def observation_model(self, state_estimate_k, z_k_observation_vector, p_k_minus_1):
        """
        Extended Kalman Filter. Fuses noisy sensor measurement to
        create an optimal estimate of the state of the robotic system.
        """

        # Predict the state covariance estimate based on the previous
        # covariance and some noise (linear covariance)
        P_k = self.A_k_minus_1 @ p_k_minus_1 @ self.A_k_minus_1.T + self.Q_k

        # Calculate the difference between the actual sensor measurements
        # at time k minus what the measurement model predicted
        # the sensor measurements would be for the current timestep k.
        measurement_residual_y_k = z_k_observation_vector - ((self.H_k @ state_estimate_k) + self.sensor_noise_w_k)

        # Calculate the measurement residual covariance
        S_k = self.H_k @ P_k @ self.H_k.T + self.R_k

        # Calculate the near-optimal Kalman gain
        # We use pseudo inverse since some of the matrices might be
        # non-square or singular.
        K_k = P_k @ self.H_k.T @ np.linalg.pinv(S_k)

        # Calculate an updated state estimate for time k
        observation_model_state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ self.H_k @ P_k)

        # Return the updated state and covariance estimates
        return observation_model_state_estimate_k, P_k


def run_EKF_traj_data(test_df):
    measurement_uncertainty = test_df['sigma_pos'][0]
    EKF_solver = EKF(sensor_measurement_uncertainty=measurement_uncertainty, update_space_process_noise=1e-5,
                     show_info=False)

    avg_meas_err = []
    avg_ekf_err = []

    avg_meas_uncertainty = []
    avg_ekf_uncertainty = []
    ekf_proposed_traj = {}
    truth_traj = defaultdict(list)
    measured_proposed_traj = defaultdict(list)
    traj_no = 0
    for index in tqdm(range(1, len(test_df))):
        t_minus_row = test_df.iloc[index - 1]
        t_current_row = test_df.iloc[index]

        if t_current_row.timestamp == 0:  # end ekf estimate
            avg_meas_err.append(np.mean(traj_meas_err))
            avg_ekf_err.append(np.mean(traj_ekf_err))

            avg_meas_uncertainty.append(np.mean(traj_meas_uncertainty))
            avg_ekf_uncertainty.append(np.mean(traj_ekf_uncertainty, axis=0))
            continue
        elif t_minus_row.timestamp == 0:  # reset ekf on new trajectory...
            traj_no += 1
            traj_meas_err = []
            traj_ekf_err = []

            traj_meas_uncertainty = []
            traj_ekf_uncertainty = []

            ekf_proposed_traj[traj_no] = defaultdict(list)

            state_estimate_k_minus_1 = np.array(
                [t_minus_row['tx_measured'], t_minus_row['ty_measured'], t_minus_row['tz_measured']])
            p_k_minus_1 = None
        else:
            state_estimate_k_minus_1 = optimal_state_estimate_k
            p_k_minus_1 = covariance_estimate_k

        # var retrieval
        truth_pos_vector = np.array([t_current_row['tx'], t_current_row['ty'], t_current_row['tz']])
        control_vector_k_minus_1 = np.array(
            [t_minus_row['vx_measured'], t_minus_row['vy_measured'], t_minus_row['vz_measured']])
        obs_vector_z_k = np.array(
            [t_current_row['tx_measured'], t_current_row['ty_measured'], t_current_row['tz_measured']])
        dt = t_current_row['timestamp'] - t_minus_row['timestamp']

        # solving
        optimal_state_estimate_k, covariance_estimate_k = EKF_solver.forward(obs_vector_z_k,
                                                                             # Most recent sensor measurement
                                                                             state_estimate_k_minus_1,
                                                                             # Our most recent estimate of the state
                                                                             control_vector_k_minus_1,
                                                                             # Our most recent control input
                                                                             dt,  # Time interval
                                                                             p_k_minus_1
                                                                             # Our most recent state covariance matrix
                                                                             )
        truth_traj[traj_no].append(truth_pos_vector)

        # for overall trajectory evaluation
        traj_ekf_err.append(np.linalg.norm(truth_pos_vector - optimal_state_estimate_k))
        traj_ekf_uncertainty.append(np.diagonal(covariance_estimate_k))
        traj_meas_err.append(np.linalg.norm(truth_pos_vector - obs_vector_z_k))
        traj_meas_uncertainty.append(t_current_row['sigma_pos'])

        # dictionary appending for per trajectory evaluation
        ekf_proposed_traj[traj_no]['state_est'].append(optimal_state_estimate_k)
        ekf_proposed_traj[traj_no]['state_uncer'].append(np.diagonal(covariance_estimate_k))

    print('Done producing estimates on test data!')

    return truth_traj, ekf_proposed_traj, avg_ekf_err, avg_ekf_uncertainty, avg_meas_err, avg_meas_uncertainty

