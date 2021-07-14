import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import h5py
from scipy.interpolate import interp1d


'''
    Input
    1. File paths of hdf5 exported from pose tracking software
    2. Frame rate of the videos analyzed
   Output: directly save kinematics csv files to the as file as the given hdf5
'''

def h5_to_csv(file_dir, frame_rate):
    save_to = os.path.abspath(os.path.join(file_dir, os.pardir))

    with h5py.File(file_dir, 'r') as f:
        dset_names = list(f.keys())
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:].transpose()
        nodes = [n.decode() for n in f["node_names"][:]]

    def fill_missing(Y, kind="cubic"):
        """Fills missing values independently along each dimension after the first"""

        initial_shape = Y.shape
        Y = Y.reshape((initial_shape[0], -1))
        # Interpolate along each side
        for i in range(Y.shape[-1]):
            y = Y[:, i]

            # Build interpolant
            x = np.flatnonzero(~np.isnan(y))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            Y[xq, i] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(Y)
            Y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[~mask])
        # Restore to initial shape
        Y = Y.reshape(initial_shape)
        return Y


    tracks_matrix = fill_missing(tracks_matrix)


    def filtered_coordinate(node_index, x_or_y, win=15, poly=3):
        """generate coordinate and smooth data with Savitzky-Golay Filter"""
        if x_or_y == 'x':
            coordinates = np.array([])
            i = 0
            while i < tracks_matrix.shape[0]:
                coordinates = np.append(coordinates, tracks_matrix[i - 1, node_index, 0, 0])
                i += 1
            return savgol_filter(coordinates, win, poly)
        if x_or_y == 'y':
            coordinates = np.array([])
            i = 0
            while i < tracks_matrix.shape[0]:
                coordinates = np.append(coordinates, tracks_matrix[i - 1, node_index, 1, 0])
                i += 1
        return savgol_filter(coordinates, win, poly)


    ihead = 0
    itorso = 1
    itailhead = 2
    head_x_coordinates = filtered_coordinate(ihead, 'x')
    head_y_coordinates = filtered_coordinate(ihead, 'y')
    torso_x_coordinates = filtered_coordinate(itorso, 'x')
    torso_y_coordinates = filtered_coordinate(itorso, 'y')
    tailhead_x_coordinates = filtered_coordinate(itailhead, 'x')
    tailhead_y_coordinates = filtered_coordinate(itailhead, 'y')


    def get_displacement(node, framenum):
        if framenum == tracks_matrix.shape[0] - 1:
            return 0
        else:
            if node == 'Head':
                xd = head_x_coordinates[framenum + 1] - head_x_coordinates[framenum]
                yd = head_y_coordinates[framenum + 1] - head_y_coordinates[framenum]
            elif node == 'Torso':
                xd = torso_x_coordinates[framenum + 1] - torso_x_coordinates[framenum]
                yd = torso_y_coordinates[framenum + 1] - torso_y_coordinates[framenum]
            elif node == 'Tailhead':
                xd = tailhead_x_coordinates[framenum + 1] - tailhead_x_coordinates[framenum]
                yd = tailhead_y_coordinates[framenum + 1] - tailhead_y_coordinates[framenum]
            return np.sqrt(xd ** 2 + yd ** 2)


    head_displacement = np.array([])
    i = 0
    for i in range(tracks_matrix.shape[0]):
        head_displacement = np.append(head_displacement, get_displacement('Head', i))
        i += 1

    torso_displacement = np.array([])
    i = 0
    for i in range(tracks_matrix.shape[0]):
        torso_displacement = np.append(torso_displacement, get_displacement('Torso', i))
        i += 1

    tailhead_displacement = np.array([])
    i = 0
    for i in range(tracks_matrix.shape[0]):
        tailhead_displacement = np.append(tailhead_displacement, get_displacement('Tailhead', i))
        i += 1


    # velocity in coordinates/second
    def get_velocity(node_displacement, sigma=3):
        raw_velocity = node_displacement / (1 / frame_rate)
        # smooth with gaussian filter
        filtered = gaussian_filter1d(raw_velocity, sigma)
        return filtered


    head_velocity = get_velocity(head_displacement)
    torso_velocity = get_velocity(torso_displacement)
    tailhead_velocity = get_velocity(tailhead_displacement)


    # acceleration in coordinate/s^2
    def get_acceleration(node_velocity):
        # make a matrix and is shifted by 1 frame
        nextframe = np.append(np.delete(node_velocity, 0), 0)
        return (nextframe - node_velocity) / (1 / frame_rate)


    head_acceleration = get_acceleration(head_velocity)
    torso_acceleration = get_acceleration(torso_velocity)
    tailhead_acceleration = get_acceleration(tailhead_velocity)


    # calculate rotation and angular velocity
    def get_coordinates(framenum, node):
        if node == 'Head':
            return np.array([head_x_coordinates[framenum - 1], head_y_coordinates[framenum - 1]])
        if node == 'Torso':
            return np.array([torso_x_coordinates[framenum - 1], torso_y_coordinates[framenum - 1]])
        if node == 'Tailhead':
            return np.array([tailhead_x_coordinates[framenum - 1], tailhead_y_coordinates[framenum - 1]])


    def rotational_angle(framenum):
        """convert coordinates to egocentric coordinates"""
        y1 = get_coordinates(framenum, 'Head')[1] - get_coordinates(framenum, 'Torso')[1]
        y2 = get_coordinates(framenum + 1, 'Head')[1] - get_coordinates(framenum, 'Torso')[1]
        x1 = get_coordinates(framenum, 'Head')[0] - get_coordinates(framenum, 'Torso')[0]
        x2 = get_coordinates(framenum + 1, 'Head')[0] - get_coordinates(framenum, 'Torso')[0]
        # if the radians is negative, it means that it is in the 3rd/4th quadrant
        if math.atan2(y1, x1) <= 0:
            x1y1 = 360 + math.degrees(math.atan2(y1, x1))
        else:
            x1y1 = math.degrees(math.atan2(y1, x1))
        if math.atan2(y2, x2) <= 0:
            x2y2 = 360 + math.degrees(math.atan2(y2, x2))
        else:
            x2y2 = math.degrees(math.atan2(y2, x2))
        return x2y2 - x1y1


    # iterate through the dataset to extract rotation angle and angular velocity

    # rotation angle
    org_rotation_angle = []
    i = 0
    while i < occupancy_matrix.shape[0]:
        org_rotation_angle.append(rotational_angle(i))
        i += 1
    org_rotation_angle = np.array(org_rotation_angle)
    # address outliers using a median filter
    # outliers come from turning at positive x-axis (e.g before:350degrees, after:5degrees, 5-350=-345)

    rotation_angle = org_rotation_angle.copy()


    def fix_outliers(Y, kind="cubic"):
        """Fills missing values independently along each dimension after the first"""
        initial_shape = Y.shape
        Y = Y.reshape((initial_shape[0], -1))
        # Interpolate along each side
        for i in range(Y.shape[-1]):
            y = Y[:, i]

            # Build interpolant
            x = np.flatnonzero(~(abs(Y) > 300))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill Outliers
            xq = np.flatnonzero(abs(Y) > 300)
            Y[xq, i] = f(xq)

            # Fill leading or trailing outliers with the nearest values
            mask = np.isnan(Y)
            Y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[~mask])
        # Restore to initial shape
        Y = Y.reshape(initial_shape)
        return Y


    filtered_rotation_angle = fix_outliers(rotation_angle)
    final_rotation_angle = gaussian_filter1d(filtered_rotation_angle, 3)

    angular_velocity = final_rotation_angle / (1 / frame_rate)
    nextframe = np.append(np.delete(angular_velocity, 0), 0)
    angular_acceleration = (nextframe - angular_velocity) / (1 / frame_rate)

    df = pd.DataFrame({'Head xCoordinates': head_x_coordinates,
                       'Head yCoordinates': head_y_coordinates,
                       'Torso xCoordinates': torso_x_coordinates,
                       'Torso yCoordinates': torso_y_coordinates,
                       'Tailhead xCoordinates': tailhead_x_coordinates,
                       'Tailhead yCoordinates': tailhead_y_coordinates,
                       'Tailhead Displacement': tailhead_displacement,
                       'Head Velocity': head_velocity,
                       'Torso Velocity': torso_velocity,
                       'Tailhead Velocity': tailhead_velocity,
                       'Head Acceleration': head_acceleration,
                       'Torso Acceleration': torso_acceleration,
                       'Tailhead Acceleration': tailhead_acceleration,
                       'Angular Velocity (degrees/sec)': angular_velocity,
                       'Angular Acceleration (degrees/sec^2)': angular_acceleration,
                       'rotation angle': final_rotation_angle})
    df.to_csv(save_to + '.csv')

