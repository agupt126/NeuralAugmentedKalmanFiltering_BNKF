import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch
def construct_dict_folds(full_trajectory_dict, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = []
    keys = list(full_trajectory_dict.keys())

    for train_indices, test_indices in kf.split(keys):
        train_keys = [keys[i] for i in train_indices]
        test_keys = [keys[i] for i in test_indices]
        indices.append((train_keys, test_keys))
    return indices


def return_index_cv_params(full_proc_traj_data_in, cv_indices):
    train_X_mats = []
    train_y_vecs = []
    test_X_mats = []
    test_y_vecs = []

    physics_metadata_train_dicts = []
    physics_metadata_test_dicts = []

    for index in cv_indices:
        train_pmd_dict = {
            'inputPos': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputPos'] for key in index[0]],
                                  dim=0),
            'inputVel': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputVel'] for key in index[0]],
                                  dim=0),
            'inputObs': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputObs'] for key in index[0]],
                                  dim=0),
            'dtVec': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['dtVec'] for key in index[0]], dim=0)}

        train_X_mats.append(torch.cat([full_proc_traj_data_in[key]['X_mat'] for key in index[0]], dim=0))
        train_y_vecs.append(torch.cat([full_proc_traj_data_in[key]['y_vec'] for key in index[0]], dim=0))
        physics_metadata_train_dicts.append(train_pmd_dict)

        test_pmd_dict = {
            'inputPos': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputPos'] for key in index[1]],
                                  dim=0),
            'inputVel': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputVel'] for key in index[1]],
                                  dim=0),
            'inputObs': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['inputObs'] for key in index[1]],
                                  dim=0),
            'dtVec': torch.cat([full_proc_traj_data_in[key]['physics_meta_data']['dtVec'] for key in index[1]], dim=0)}

        test_X_mats.append(torch.cat([full_proc_traj_data_in[key]['X_mat'] for key in index[1]], dim=0))
        test_y_vecs.append(torch.cat([full_proc_traj_data_in[key]['y_vec'] for key in index[1]], dim=0))
        physics_metadata_test_dicts.append(test_pmd_dict)

    return train_X_mats, train_y_vecs, test_X_mats, test_y_vecs, physics_metadata_train_dicts, physics_metadata_test_dicts


def convert_std_list_to_matrix(std_vec):
    variances = []
    for std in std_vec:
        variances.append(std**2)
    return np.diag(np.array(variances))

def store_truth_data(aDF):
    truth_data = {}
    truth_data['timestamps'] = np.array(aDF['timestamp'])
    truth_data['x'] = np.array(aDF['tx'])
    truth_data['y'] = np.array(aDF['ty'])
    truth_data['z'] = np.array(aDF['tz'])
    truth_data['vx'] = np.array(aDF['vx'])
    truth_data['vy'] = np.array(aDF['vy'])
    truth_data['vz'] = np.array(aDF['vz'])
    return truth_data


def visualize_3d_scatter(x_1, y_1, z_1, label_1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_1, y_1, z_1, c='g', marker='x', label=label_1)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Truth Trajectory')
    ax.legend()
    plt.show()
    return None


def visualize_dual_3d_scatter(x_1, y_1, z_1, label_1, x_2, y_2, z_2, label_2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_1, y_1, z_1, c='g', marker='x', s=100, label=label_1)
    ax.plot(x_2, y_2, z_2, c='g', marker='o', label=label_2)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Truth Trajectory')
    ax.legend()
    plt.show()
    return None
