import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from helper.data_handler import TrajectoryData

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def evaluate_bayesian_neural_model(a_model, clean_df, test_mat_X, test_mat_y, z_obs, pinn_eval=False, apply_EKF_in=False, continuous_eval=False, num_samples=10):
    avg_model_err = []
    avg_model_uncertainty = []
    model_proposed_traj = defaultdict(list)
    truth_traj = defaultdict(list)

    traj_no = 0
    for index in tqdm(range(1, len(test_mat_X))):
        t_minus_row = test_mat_X[index - 1]
        row_df = clean_df.iloc[index - 1]

        if clean_df.iloc[index].timestamp == 0:
            avg_model_err.append(np.mean(traj_model_err))
            avg_model_uncertainty.append(np.mean(traj_model_uncertainty))
            continue
        elif clean_df.iloc[index - 1].timestamp == 0:
            traj_no += 1
            traj_model_err = []
            traj_model_uncertainty = []
            model_proposed_traj[traj_no] = defaultdict(list)

        truth_pos_vector = np.array([test_mat_y[index, 0], test_mat_y[index, 1], test_mat_y[index, 2]])
        if pinn_eval:

            v_meas = torch.tensor(row_df[['vx_measured', 'vy_measured', 'vz_measured']].to_numpy(),
                                  dtype=torch.float32)
            dt_meas = torch.tensor(row_df[['delta_time']].to_numpy(), dtype=torch.float32)

            if continuous_eval and len(traj_model_err) > 0: # if model state_estimate has been produced..
                x_meas = torch.tensor(model_state_estimate, dtype=torch.float32)
            else:
                x_meas = torch.tensor(row_df[['tx_measured', 'ty_measured', 'tz_measured']].to_numpy(),
                                      dtype=torch.float32)

            full_X = torch.tensor(row_df[['tx_measured_plus', 'ty_measured_plus', 'tz_measured_plus', 'sigma_pos', 'sigma_vel']].to_numpy(),
                                  dtype=torch.float32)

            model_state_estimate, uncer = a_model.predict(x_meas, v_meas, dt_meas, full_X, z_obs[index], num_samples, apply_ekf=apply_EKF_in)
        else:
            model_state_estimate, uncer = a_model.predict(t_minus_row, num_samples)
            model_state_estimate = model_state_estimate.numpy()

        # for a quick overall traj evaluation
        traj_model_err.append(np.linalg.norm(truth_pos_vector - model_state_estimate))
        traj_model_uncertainty.append(uncer)

        # for a per traj evaluation
        model_proposed_traj[traj_no]['state_est'].append(model_state_estimate)
        model_proposed_traj[traj_no]['state_uncer'].append(uncer)

        if index == len(test_mat_X) - 1 and len(avg_model_err) == 0:  # for the case of only 1 test trajectory..
            avg_model_err.append(np.mean(traj_model_err))
            avg_model_uncertainty.append(np.mean(traj_model_uncertainty))

    return avg_model_err, avg_model_uncertainty, model_proposed_traj


import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def visualize_final_estimates_traj(in_traj_no, truth_traj_data, ekf_traj_data, bnn_traj_data, pinn_traj_data, threeD=False,
                                     smoothing=False):
    """
        TRUTH: RED
        EKF: YELLOW
        BNN: BLUE
        PINN: MAGENTA
    """

    truth_data = truth_traj_data[in_traj_no]
    ekf_data = ekf_traj_data[in_traj_no]['state_est']
    ekf_uncer = ekf_traj_data[in_traj_no]['state_uncer']
    bnn_data = bnn_traj_data[in_traj_no]['state_est']
    bnn_uncer = bnn_traj_data[in_traj_no]['state_uncer']
    pinn_data = pinn_traj_data[in_traj_no]['state_est']
    pinn_uncer = pinn_traj_data[in_traj_no]['state_uncer']

    x_coords_truth = [point[0] for point in truth_data]
    y_coords_truth = [point[1] for point in truth_data]
    z_coords_truth = [point[2] for point in truth_data]

    x_coords_ekf = [point[0] for point in ekf_data]
    y_coords_ekf = [point[1] for point in ekf_data]
    z_coords_ekf = [point[2] for point in ekf_data]

    x_coords_bnn = [point[0] for point in bnn_data]
    y_coords_bnn = [point[1] for point in bnn_data]
    z_coords_bnn = [point[2] for point in bnn_data]

    x_coords_pinn = [point[0] for point in pinn_data]
    y_coords_pinn = [point[1] for point in pinn_data]
    z_coords_pinn = [point[2] for point in pinn_data]

    ## Error + Uncertainty Plot ##
    ekf_euclid_errs = np.array([np.linalg.norm(a - b) for a, b in zip(truth_data, ekf_data)])
    ekf_uncer_mean = np.array([np.sqrt(np.mean(vector)) for vector in ekf_uncer])
    ekf_lower_bound = ekf_euclid_errs - ekf_uncer_mean
    ekf_upper_bound = ekf_euclid_errs + ekf_uncer_mean

    bnn_euclid_errs = np.array([np.linalg.norm(a - b) for a, b in zip(truth_data, bnn_data)])
    bnn_uncer_mean = np.array([np.mean(vector.numpy()) for vector in bnn_uncer])
    bnn_lower_bound = bnn_euclid_errs - bnn_uncer_mean
    bnn_upper_bound = bnn_euclid_errs + bnn_uncer_mean

    pinn_euclid_errs = np.array([np.linalg.norm(a - b) for a, b in zip(truth_data, pinn_data)])
    pinn_uncer_mean = np.array([np.mean(vector) for vector in pinn_uncer])
    pinn_lower_bound = pinn_euclid_errs - pinn_uncer_mean
    pinn_upper_bound = pinn_euclid_errs + pinn_uncer_mean

    plt.figure(figsize=(10, 6))

    plt.plot(ekf_euclid_errs, label='EKF', color='y', linewidth=3)
    plt.fill_between(range(len(ekf_euclid_errs)), ekf_lower_bound, ekf_upper_bound, color='y', alpha=0.3,
                     label='EKF Uncertainty')

    plt.plot(moving_average(bnn_euclid_errs, 5), label='BNN', color='b', linewidth=3)
    plt.fill_between(range(len(bnn_euclid_errs)), bnn_lower_bound, bnn_upper_bound, color='b', alpha=0.3,
                     label='BNN Uncertainty')

    plt.plot(moving_average(pinn_euclid_errs, 5), label='PINN', color='m', linewidth=3)
    plt.fill_between(range(len(pinn_euclid_errs)), pinn_lower_bound, pinn_upper_bound, color='m', alpha=0.3,
                     label='PINN Uncertainty')

    plt.title('Model Euclidean Error wrt Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Error [m]')
    plt.legend()
    plt.show()
    ## Error + Uncertainty Plot ##

    ## Trajectory Generation Compare ##
    if smoothing:
        window_size = int(0.05 * len(x_coords_bnn))
        x_coords_bnn = moving_average(x_coords_bnn, window_size)
        y_coords_bnn = moving_average(y_coords_bnn, window_size)
        z_coords_bnn = moving_average(z_coords_bnn, window_size)

        x_coords_pinn = moving_average(x_coords_pinn, window_size)
        y_coords_pinn = moving_average(y_coords_pinn, window_size)
        z_coords_pinn = moving_average(z_coords_pinn, window_size)

    if threeD:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_coords_truth, y_coords_truth, z_coords_truth, c='r', marker='x', label="Truth")
        ax.scatter(x_coords_ekf, y_coords_ekf, z_coords_ekf, c='y', marker='^', label="EKF")
        ax.scatter(x_coords_bnn, y_coords_bnn, z_coords_bnn, c='b', marker='o', label="BNN")
        ax.scatter(x_coords_pinn, y_coords_pinn, z_coords_pinn, c='m', marker='*', label="PINN")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        ax.set_title(f'Model Approximations of Truth Trajectory {in_traj_no}')
        ax.legend()
        plt.show()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # X
        axs[0].plot(x_coords_truth, label='Truth', color='r', marker='x', linestyle='-', markersize=5)
        axs[0].plot(x_coords_ekf, label='EKF', color='y', marker='^', linestyle='-', markersize=5)
        axs[0].plot(x_coords_bnn, label='BNN', color='b', marker='o', linestyle='-', markersize=5)
        axs[0].plot(x_coords_pinn, label='PINN', color='m', marker='*', linestyle='-', markersize=5)
        axs[0].set_title(f'X Approximations of Truth Trajectory {in_traj_no}')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('X [m]')
        axs[0].legend()
        axs[0].grid(True)

        # Y
        axs[1].plot(y_coords_truth, label='Truth', color='r', marker='x', linestyle='-', markersize=5)
        axs[1].plot(y_coords_ekf, label='EKF', color='y', marker='^', linestyle='-', markersize=5)
        axs[1].plot(y_coords_bnn, label='BNN', color='b', marker='o', linestyle='-', markersize=5)
        axs[1].plot(y_coords_pinn, label='PINN', color='m', marker='*', linestyle='-', markersize=5)
        axs[1].set_title(f'Y Approximations of Truth Trajectory {in_traj_no}')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Y [m]')
        axs[1].legend()
        axs[1].grid(True)

        # Z
        axs[2].plot(z_coords_truth, label='Truth', color='r', marker='x', linestyle='-', markersize=5)
        axs[2].plot(z_coords_ekf, label='EKF', color='y', marker='^', linestyle='-', markersize=5)
        axs[2].plot(z_coords_bnn, label='BNN', color='b', marker='o', linestyle='-', markersize=5)
        axs[2].plot(z_coords_pinn, label='PINN', color='m', marker='*', linestyle='-', markersize=5)
        axs[2].set_title(f'Z Approximations of Truth Trajectory {in_traj_no}')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Z [m]')
        axs[2].legend()
        axs[2].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()
    ## Trajectory Generation Compare ##
    return None
