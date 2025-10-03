from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import pandas as pd
import matplotlib as mpl

class Trajectory:
    def __init__(self, truth_data, traj_no, measurement_only_eval=False):
        self.traj_name = f"Trajectory-{traj_no}"
        self.measurement_only_eval = measurement_only_eval
        self.truth_time_data = truth_data['timestamps']
        self.truth_x = truth_data['x']
        self.truth_y = truth_data['y']
        self.truth_z = truth_data['z']
        self.truth_vx = truth_data['vx']
        self.truth_vy = truth_data['vy']
        self.truth_vz = truth_data['vz']
        self.start_time = datetime.utcfromtimestamp(float(truth_data['timestamps'][0]))
        self.traj_length = len(truth_data['timestamps'])

        self.orig_truth_time_data = truth_data['timestamps']
        self.orig_truth_x = truth_data['x']
        self.orig_truth_y = truth_data['y']
        self.orig_truth_z = truth_data['z']
        self.orig_truth_vx = truth_data['vx']
        self.orig_truth_vy = truth_data['vy']
        self.orig_truth_vz = truth_data['vz']
        self.orig_start_time = datetime.utcfromtimestamp(float(truth_data['timestamps'][0]))
        self.orig_traj_length = len(truth_data['timestamps'])

        self.truth_time_data_filtered_indices = None

        self.__construct_stonesoup_truth()

        self.measured_x = None
        self.measured_y = None
        self.measured_z = None
        self.measured_vx = None
        self.measured_vy = None
        self.measured_vz = None
        self.sigma_range = None
        self.sigma_range_rate = None
        self.sigma_elevation = None
        self.sigma_bearing = None
        self.vel = False
        self.remove_pct = None

    def __construct_stonesoup_truth(self):
        self.truth_state_traj = GroundTruthPath([GroundTruthState(
            [self.truth_x[0], self.truth_vx[0], self.truth_y[0], self.truth_vy[0], self.truth_z[0], self.truth_vz[0]],
            timestamp=self.start_time)])

        for k in range(1, self.traj_length):
            self.truth_state_traj.append(GroundTruthState(
                [self.truth_x[k], self.truth_vx[k], self.truth_y[k], self.truth_vy[k], self.truth_z[k],
                 self.truth_vz[k]], timestamp=self.start_time + timedelta(seconds=float(self.truth_time_data[k]))))

    def __introduce_gaps(self, arr, remove_pct=25):
        N = len(arr)
        keep_pct = 100 - remove_pct
        chunk_size = round(N / 10)

        keep = round(chunk_size * (keep_pct / 100))
        remove = round(chunk_size * (remove_pct / 100))
        if keep == 0:
            raise ValueError("Calculated keep value is too small, increase keep percentage or chunk size.")

        filtered_list = []
        removed_indices = []
        for i in range(0, N, keep + remove):
            filtered_list.extend(arr[i:i + keep])
            removed_indices.extend(range(i + keep, min(i + keep + remove, N)))

        self.truth_time_data_filtered_indices = np.array(removed_indices)
        return np.array(filtered_list)

    def apply_sampling_gaps(self, remove_pct=25):
        self.remove_pct = remove_pct
        self.truth_time_data = self.__introduce_gaps(self.truth_time_data, remove_pct)
        self.truth_x = self.__introduce_gaps(self.truth_x, remove_pct)
        self.truth_y = self.__introduce_gaps(self.truth_y, remove_pct)
        self.truth_z = self.__introduce_gaps(self.truth_z, remove_pct)
        self.truth_vx = self.__introduce_gaps(self.truth_vx, remove_pct)
        self.truth_vy = self.__introduce_gaps(self.truth_vy, remove_pct)
        self.truth_vz = self.__introduce_gaps(self.truth_vz, remove_pct)

        self.start_time = datetime.utcfromtimestamp(float(self.truth_time_data[0]))
        self.traj_length = len(self.truth_time_data)
        if self.measurement_only_eval:
            self.__construct_stonesoup_truth()
        return None

    def set_converted_measurement_data(self, measurement_data, vel=False):
        assert len(measurement_data['x']) == len(measurement_data['y']) == len(
            measurement_data['z'])

        if len(self.truth_time_data_filtered_indices) == 0 or self.measurement_only_eval:
            self.measured_x = np.array(measurement_data['x']).ravel()
            self.measured_y = np.array(measurement_data['y']).ravel()
            self.measured_z = np.array(measurement_data['z']).ravel()

            if vel:
                self.measured_vx = np.array(measurement_data['vx']).ravel()
                self.measured_vy = np.array(measurement_data['vy']).ravel()
                self.measured_vz = np.array(measurement_data['vz']).ravel()

        else:
            self.measured_x = np.delete(np.array(measurement_data['x']).ravel(), self.truth_time_data_filtered_indices)
            self.measured_y = np.delete(np.array(measurement_data['y']).ravel(), self.truth_time_data_filtered_indices)
            self.measured_z = np.delete(np.array(measurement_data['z']).ravel(), self.truth_time_data_filtered_indices)

            if vel:
                self.measured_vx = np.delete(np.array(measurement_data['vx']).ravel(), self.truth_time_data_filtered_indices)
                self.measured_vy = np.delete(np.array(measurement_data['vy']).ravel(), self.truth_time_data_filtered_indices)
                self.measured_vz = np.delete(np.array(measurement_data['vz']).ravel(), self.truth_time_data_filtered_indices)

        self.vel = vel
        return None

    def set_measurement_sigmas(self, sigma_elevation, sigma_bearing, sigma_range, sigma_range_rate):
        self.sigma_elevation = np.full(self.traj_length, sigma_elevation)
        self.sigma_bearing = np.full(self.traj_length, sigma_bearing)
        self.sigma_range = np.full(self.traj_length, sigma_range)
        self.sigma_range_rate = np.full(self.traj_length, sigma_range_rate)
        return None

    def return_trajectory_df(self):
        if self.vel:
            data = {
                "timestamp": self.truth_time_data,
                "truth_x": self.truth_x,
                "truth_y": self.truth_y,
                "truth_z": self.truth_z,
                "truth_vx": self.truth_vx,
                "truth_vy": self.truth_vy,
                "truth_vz": self.truth_vz,
                "measured_x": self.measured_x,
                "measured_y": self.measured_y,
                "measured_z": self.measured_z,
                "measured_vx": self.measured_vx,
                "measured_vy": self.measured_vy,
                "measured_vz": self.measured_vz,
            }
        else:
            data = {
                "timestamp": self.truth_time_data,
                "truth_x": self.truth_x,
                "truth_y": self.truth_y,
                "truth_z": self.truth_z,
                "truth_vx": self.truth_vx,
                "truth_vy": self.truth_vy,
                "truth_vz": self.truth_vz,
                "measured_x": self.measured_x,
                "measured_y": self.measured_y,
                "measured_z": self.measured_z,
            }
        return pd.DataFrame(data)

    def return_nn_prepared_df(self, aDF):
        # shift processing
        altered_df = aDF.copy()
        altered_df['delta_time'] = aDF['timestamp'].diff().shift(-1)
        altered_df.fillna({'delta_time': 0}, inplace=True)
        altered_df['measured_plus_x'] = aDF['measured_x'].shift(-1, fill_value=0)
        altered_df['measured_plus_y'] = aDF['measured_y'].shift(-1, fill_value=0)
        altered_df['measured_plus_z'] = aDF['measured_z'].shift(-1, fill_value=0)

        altered_df['sigma_range'] = self.sigma_range
        altered_df['sigma_range_rate'] = self.sigma_range_rate
        altered_df['sigma_elevation'] = self.sigma_elevation
        altered_df['sigma_bearing'] = self.sigma_bearing

        if self.vel:
            altered_df['measured_plus_vx'] = aDF['measured_vx'].shift(-1, fill_value=0)
            altered_df['measured_plus_vy'] = aDF['measured_vy'].shift(-1, fill_value=0)
            altered_df['measured_plus_vz'] = aDF['measured_vz'].shift(-1, fill_value=0)
        return altered_df

    def visualize_3d_scatter(self, orig=False, save_fig=False):
        # Set IEEE-friendly font sizes
        mpl.rcParams.update({
            "font.size": 9,  # base font
            "axes.titlesize": 10,  # subplot title
            "axes.labelsize": 9,  # axis labels
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300  # crisp for publication
        })

        # IEEE double-column width (~7.16 in), square aspect for 3D
        fig = plt.figure(figsize=(7.16, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        if orig:
            ax.scatter(self.orig_truth_x, self.orig_truth_y, self.orig_truth_z,
                       c='g', marker='x', label="Truth")
        else:
            ax.scatter(self.truth_x, self.truth_y, self.truth_z,
                       c='g', marker='x', label="Truth")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')


        plt.tight_layout()
        if save_fig:
            plt.savefig(f'../../report/images/truth_traj_3d_{self.remove_pct}_SR.png',
                        dpi=600, bbox_inches='tight')  # high-res
        plt.show()
        return None

    def visualize_3d_trajectory(self, save_gif=False):
        def create_animation(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            line, = ax.plot([], [], [], lw=2)

            ax.set_xlim(self.truth_x.min(), self.truth_x.max())
            ax.set_ylim(self.truth_y.min(), self.truth_y.max())
            ax.set_zlim(self.truth_z.min(), self.truth_z.max())

            ax.set_xlabel('X Pos')
            ax.set_ylabel('Y Pos')
            ax.set_zlabel('Z Pos')

            def init():
                line.set_data([], [])
                line.set_3d_properties([])
                return line,

            def update(frame):
                line.set_data(self.trajectory_df['tx'][:frame], self.trajectory_df['ty'][:frame])
                line.set_3d_properties(self.trajectory_df['tz'][:frame])
                return line,

            if save_gif:
                ani = FuncAnimation(fig, update, frames=len(self.trajectory_df), init_func=init, blit=True,
                                    interval=100)
                ani.save(self.gif_path, writer='pillow', fps=30)

            plt.show()

    def visualize_truth_versus_measurement(self):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(self.orig_truth_time_data[1:], self.orig_truth_x[1:], label="Truth", color="g")
        axes[0].plot(self.truth_time_data[1:], self.measured_x[1:], label="Radar Measurement", color="b")
        axes[0].set_xlabel("Timestamp [s]")
        axes[0].set_ylabel("Position [m]")
        axes[0].set_title("X: Cartesian Datapoints")
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(self.orig_truth_time_data[1:], self.orig_truth_y[1:], label="Truth", color="g")
        axes[1].plot(self.truth_time_data[1:], self.measured_y[1:], label="Radar Measurement", color="b")
        axes[1].set_xlabel("Timestamp [s]")
        axes[1].set_ylabel("Position [m]")
        axes[1].set_title("Y: Cartesian Datapoints")
        axes[1].grid()
        axes[1].legend()

        axes[2].plot(self.orig_truth_time_data[1:], self.orig_truth_z[1:], label="Truth", color="g")
        axes[2].plot(self.truth_time_data[1:], self.measured_z[1:], label="Radar Measurement", color="b")
        axes[2].set_xlabel("Timestamp [s]")
        axes[2].set_ylabel("Position [m]")
        axes[2].set_title("Z: Cartesian Datapoints")
        axes[2].grid()
        axes[2].legend()
        plt.tight_layout()
        plt.show()

    def visualize_truth_versus_filter(self, filter_state_data, filter_uncer_data):
        if self.measurement_only_eval:
            time_values = self.truth_time_data[1:]
        else:
            time_values = self.orig_truth_time_data[1:]

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(self.orig_truth_time_data[1:], self.orig_truth_x[1:], label="Truth", color="g")
        axes[0].plot(time_values, filter_state_data['x'], label="Filter", color="r")
        axes[0].errorbar(time_values, filter_state_data['x'], yerr=filter_uncer_data['x'], fmt='o', color="r", alpha=0.2, label="Filter Uncertainty", capsize=3)
        axes[0].set_xlabel("Timestamp [s]")
        axes[0].set_ylabel("Position [m]")
        axes[0].set_title("X: Cartesian Datapoints")
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(self.orig_truth_time_data[1:], self.orig_truth_y[1:], label="Truth", color="g")
        axes[1].plot(time_values, filter_state_data['y'], label="Filter", color="r")
        axes[1].errorbar(time_values, filter_state_data['y'], yerr=filter_uncer_data['y'], fmt='o', color="r", alpha=0.2, label="Filter Uncertainty", capsize=3)
        axes[1].set_xlabel("Timestamp [s]")
        axes[1].set_ylabel("Position [m]")
        axes[1].set_title("Y: Cartesian Datapoints")
        axes[1].grid()
        axes[1].legend()

        axes[2].plot(self.orig_truth_time_data[1:], self.orig_truth_z[1:], label="Truth", color="g")
        axes[2].plot(time_values, filter_state_data['z'], label="Filter", color="r")
        axes[2].errorbar(time_values, filter_state_data['z'], yerr=filter_uncer_data['z'], fmt='o', color="r", alpha=0.2, label="Filter Uncertainty", capsize=3)
        axes[2].set_xlabel("Timestamp [s]")
        axes[2].set_ylabel("Position [m]")
        axes[2].set_title("Z: Cartesian Datapoints")
        axes[2].grid()
        axes[2].legend()
        plt.tight_layout()
        plt.show()

    def visualize_truth_versus_filter_line_bounds(self, filter_state_data, filter_uncer_data, sigma=1):
        if self.measurement_only_eval:
            time_values = self.truth_time_data[1:]
        else:
            time_values = self.orig_truth_time_data[1:]

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        # --- X Axis ---
        axes[0].plot(self.orig_truth_time_data[1:], self.orig_truth_x[1:], label="Truth", color="g")
        axes[0].plot(time_values, filter_state_data['x'], label="Filter", color="r")
        axes[0].plot(time_values, np.array(filter_state_data['x']) + sigma*np.array(filter_uncer_data['x']), linestyle='--', color='r', alpha=0.5, label=f"±{sigma}σ Bound")
        axes[0].plot(time_values, np.array(filter_state_data['x']) - sigma*np.array(filter_uncer_data['x']), linestyle='--', color='r', alpha=0.5)
        axes[0].set_xlabel("Timestamp [s]")
        axes[0].set_ylabel("Position [m]")
        axes[0].set_title("X: Cartesian Datapoints")
        axes[0].grid()
        axes[0].legend()

        # --- Y Axis ---
        axes[1].plot(self.orig_truth_time_data[1:], self.orig_truth_y[1:], label="Truth", color="g")
        axes[1].plot(time_values, filter_state_data['y'], label="Filter", color="r")
        axes[1].plot(time_values, np.array(filter_state_data['y']) + sigma*np.array(filter_uncer_data['y']), linestyle='--', color='r', alpha=0.5, label=f"±{sigma}σ Bound")
        axes[1].plot(time_values, np.array(filter_state_data['y']) - sigma*np.array(filter_uncer_data['y']), linestyle='--', color='r', alpha=0.5)
        axes[1].set_xlabel("Timestamp [s]")
        axes[1].set_ylabel("Position [m]")
        axes[1].set_title("Y: Cartesian Datapoints")
        axes[1].grid()
        axes[1].legend()

        # --- Z Axis ---
        axes[2].plot(self.orig_truth_time_data[1:], self.orig_truth_z[1:], label="Truth", color="g")
        axes[2].plot(time_values, filter_state_data['z'], label="Filter", color="r")
        axes[2].plot(time_values, np.array(filter_state_data['z']) + sigma*np.array(filter_uncer_data['z']), linestyle='--', color='r', alpha=0.5, label=f"±{sigma}σ Bound")
        axes[2].plot(time_values, np.array(filter_state_data['z']) - sigma*np.array(filter_uncer_data['z']), linestyle='--', color='r', alpha=0.5)
        axes[2].set_xlabel("Timestamp [s]")
        axes[2].set_ylabel("Position [m]")
        axes[2].set_title("Z: Cartesian Datapoints")
        axes[2].grid()
        axes[2].legend()

        plt.tight_layout()
        plt.savefig('../../report/images/ekf_traj_pred.png', dpi=300)
        plt.show()
