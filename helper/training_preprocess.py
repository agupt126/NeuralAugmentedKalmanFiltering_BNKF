import os
import torch
import pandas as pd
from helper.data_handler import TrajectoryReader
# USER DEFINITION #
relative_dataset_path_in = '../../dataset'
data_folder_path = f'{relative_dataset_path_in}/Synthetic-UAV-Flight-Trajectories'

# EXPERIMENTATION DEFINITION #
x_columns_of_interest = ['tx_measured', 'ty_measured', 'tz_measured', 'vx_measured', 'vy_measured', 'vz_measured', 'tx_measured_plus', 'ty_measured_plus', 'tz_measured_plus', 'sigma_pos', 'sigma_vel', 'delta_time']
y_columns_of_interest = ['tx', 'ty', 'tz']
z_obs_columns = ['tx_measured', 'ty_measured', 'tz_measured']
x_meas_columns = ['tx_measured', 'ty_measured', 'tz_measured']
v_meas_columns = ['vx_measured', 'vy_measured', 'vz_measured']
additional_ml_features = ['tx_measured_plus', 'ty_measured_plus', 'tz_measured_plus', 'sigma_pos', 'sigma_vel']
delta_t_meas_columns = ['delta_time']

def export_preproc_var_columns():
    columns_of_interest_dict = {
                           'x_columns_of_interest': x_columns_of_interest,
                           'y_columns_of_interest': y_columns_of_interest,
                           'x_meas_columns': x_meas_columns,
                           'v_meas_columns': v_meas_columns,
                           'delta_t_meas_columns': delta_t_meas_columns,
                           'z_obs_columns': z_obs_columns,
                           'additional_ml_features': additional_ml_features
                           }
    return columns_of_interest_dict

def readin_dataframes(train_split=0.9, sensor_pos_uncer=1, ekf=False): # [m]
    total_trajectory_count = len([file for file in os.listdir(data_folder_path) if file.endswith('.csv')])
    train_count = int(total_trajectory_count * train_split)

    train_traj_data = {}
    test_traj_data = {}
    test_split = 1 - train_split

    curr_traj_no = 0
    num_train_traj = 0
    num_test_traj = 0
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.csv'):  # Check if the file has a .csv extension
            file_path = os.path.join(data_folder_path, file_name)
            if os.path.isfile(file_path):
                tr = TrajectoryReader(file_name, relative_dataset_path=relative_dataset_path_in, sigma_pos=sensor_pos_uncer, sigma_vel=sensor_pos_uncer/1000)
                df_temp = tr.trajectory_df
                if curr_traj_no <= train_count:
                    num_train_traj += 1

                    if ekf:
                        train_traj_data[num_train_traj] = df_temp
                    else:
                        train_traj_data[num_train_traj] = preprocess_and_remove_inter_trajectory_indices(df_temp)
                else:
                    num_test_traj += 1

                    if ekf:
                        test_traj_data[num_test_traj] = df_temp
                    else:
                        test_traj_data[num_test_traj] = preprocess_and_remove_inter_trajectory_indices(df_temp)
                curr_traj_no += 1
        else:
            continue

    print(f"Number of Train Trajectories: {num_train_traj}")
    print(f"Number of Test Trajectories: {num_test_traj}")
    return train_traj_data, test_traj_data

def readin_all_dataframes(sensor_pos_uncer=1, ekf=False): # [m]
    total_trajectory_count = len([file for file in os.listdir(data_folder_path) if file.endswith('.csv')])
    full_traj_data = {}
    curr_traj_no = 0

    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.csv'):  # Check if the file has a .csv extension
            file_path = os.path.join(data_folder_path, file_name)
            if os.path.isfile(file_path):
                tr = TrajectoryReader(file_name, relative_dataset_path=relative_dataset_path_in, sigma_pos=sensor_pos_uncer, sigma_vel=sensor_pos_uncer/1000)
                df_temp = tr.trajectory_df
                if ekf:
                    full_traj_data[curr_traj_no] = df_temp
                else:
                    full_traj_data[curr_traj_no] = preprocess_and_remove_inter_trajectory_indices(df_temp)

                curr_traj_no += 1
        else:
            continue
    return full_traj_data

def preprocess_and_remove_inter_trajectory_indices(df):
    # need to ensure that there is no inter trajectory fitting going on
    zero_indices = df.index[df['timestamp'] == 0]
    adjusted_indices = [i - 1 for i in zero_indices if i - 1 >= 0]
    df_cleaned = df.drop(index=adjusted_indices)
    return df_cleaned
