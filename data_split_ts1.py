# script for train

import numpy as np
import os
import h5py
from scipy.spatial import KDTree
import multiprocessing as mp
import logging
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Tile processing parameters
compute_normals_flag = True
compute_curvature_flag = True
normals_adjusted = True
filter_small_categories = False  # todo
include_intensity = True
include_reflectance = True
min_points = 1  # todo


# Setup basic configuration for logging
# Disable logging by setting the level to CRITICAL (only critical errors will be logged)
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_processing_split_red_2m_buffer_wofilter_reflectance_refined_ts1.log',
                    filemode='w')

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename='data_processing_split_open_2m_buffer_wofilter_reflectance_refined_ts1.log',
#                     filemode='w')  # Use 'a' for append mode

# Example of logging
logging.info("Started the script.")

# Set up paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_5cm.csv')
# data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_1cm.csv') #todo
# data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm_refined.txt')
# data_file = os.path.join(BASE_DIR, 'scene_YellowArea_merged_2cm_refined.txt')
# data_file = os.path.join(BASE_DIR, 'scene_YellowArea_merged_2cm.txt')
data_file = os.path.join(BASE_DIR, 'scene_RedArea_merged_2cm_refined.txt')
# train_data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_1cm.csv')
train_data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm.csv')
# CSV Label Counts: {0.0: 422197, 1.0: 1157999, 2.0: 1217007, 3.0: 137219, 4.0: 700834, 5.0: 5539, 6.0: 184051, 7.0: 321450, 8.0: 45388, 9.0: 7073, 10.0: 38276, 11.0: 11187, 12.0: 1227, 13.0: 423, 14.0: 4594, 15.0: 1238, 16.0: 1147}
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined_ts1.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_YellowArea_2cm_val_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined_ts1.h5')
data_path_out = os.path.join(BASE_DIR,
                             'scene_RedArea_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_withfilter_reflectance_refined_ts1.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance.h5')

if not os.path.exists(os.path.dirname(data_path_out)):
    os.makedirs(os.path.dirname(data_path_out))

# Load data
data_all = np.loadtxt(data_file, delimiter=' ', skiprows=1)  # todo txt
# data_all = np.loadtxt(data_file, delimiter=',', skiprows=1)
if include_intensity:
    if include_reflectance:
        data_all = np.hstack((data_all[:, :3], data_all[:, 4][:, np.newaxis], data_all[:, -2:]))
    else:
        data_all = np.hstack((data_all[:, :3], data_all[:, -2:]))
    # Check for NaN values in the intensity column
    if np.isnan(data_all[:, -2]).any():
        logging.warning("NaN values found in the intensity column.")
        # Optionally: remove rows with NaN in intensity
        data_all = data_all[~np.isnan(data_all[:, -2])]
        logging.info(f"Filtered NaN values. Data shape after filtering: {data_all.shape}")
else:
    data_all = np.hstack((data_all[:, :3], data_all[:, -1][:, np.newaxis]))


def compute_normals(point_cloud, k=30):
    logging.info("Started computing normals.")
    logging.info(f"Data shape: {point_cloud.shape}")
    start_time = time.time()  # Start timing
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
    _, indices = nbrs.kneighbors(point_cloud)
    normals = np.zeros(point_cloud.shape, dtype=np.float64)

    for i, neighbors in enumerate(indices):
        points = point_cloud[neighbors]
        cov_matrix = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]
        normals[i] = normal

    end_time = time.time()  # End timing
    logging.info(f"Normal computing completed in {end_time - start_time:.2f} seconds.")

    return normals


def detect_major_planes(point_cloud):
    """Detect major planes (e.g., ceiling and ground) in the point cloud."""
    pca = PCA(n_components=1)
    labels = DBSCAN(eps=0.0301, min_samples=30).fit_predict(point_cloud)  # todo: k-distance graph
    major_planes = []

    for label in np.unique(labels):
        if label == -1:
            continue
        points = point_cloud[labels == label]
        pca.fit(points)
        normal = pca.components_[0]
        major_planes.append((label, normal))

    return labels, major_planes


def adjust_normals(point_cloud, normals, labels, major_planes):
    """Adjust normals based on detected planes to ensure consistent direction."""
    adjusted_normals = normals.copy()

    for label, normal in major_planes:
        points_idx = np.where(labels == label)[0]
        reference_normal = normal

        # Ensure all normals in the same plane point in the same direction
        for i in points_idx:
            if np.dot(normals[i], reference_normal) < 0:  # Check if normal is in the opposite direction
                adjusted_normals[i] = -normals[i]  # Invert the normal to align with the reference direction

    return adjusted_normals


def compute_normals_adjusted(point_cloud, k=30):  # todo
    """Compute and adjust normals for point cloud data."""
    # Step 1: Compute initial normals
    normals = compute_normals(point_cloud, k)

    # Step 2: Detect major planes (e.g., ceiling and ground)
    labels, major_planes = detect_major_planes(point_cloud)

    # Step 3: Adjust normals based on detected planes
    adjusted_normals = adjust_normals(point_cloud, normals, labels, major_planes)

    return adjusted_normals


def compute_curvature(point_cloud, k=30):
    """
    Compute local curvature for each point in the point cloud.
    Curvature is based on the ratio of the smallest eigenvalue to the sum of all eigenvalues of
    the covariance matrix of neighboring points.
    """
    logging.info("Started computing curvature.")
    logging.info(f"Data shape: {point_cloud.shape}")

    # Find k-nearest neighbors for each point in the point cloud
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
    _, indices = nbrs.kneighbors(point_cloud)

    # Initialize an array to store curvature values
    curvature = np.zeros(point_cloud.shape[0], dtype=np.float64)

    # Loop through each point and its neighbors to compute curvature
    for i, neighbors in enumerate(indices):
        points = point_cloud[neighbors]

        # Compute covariance matrix of the neighboring points
        cov_matrix = np.cov(points, rowvar=False)

        # Compute eigenvalues of the covariance matrix
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        # Curvature is the ratio of the smallest eigenvalue to the sum of all eigenvalues
        curvature[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)  # Add small value to avoid division by zero

    logging.info("Curvature computation completed.")
    return curvature


# Compute normals for the entire dataset
if compute_normals_flag:
    if normals_adjusted:
        normals = compute_normals_adjusted(data_all[:, :3])
    else:
        normals = compute_normals(data_all[:, :3])
    if compute_curvature_flag:
        curvature = compute_curvature(data_all[:, :3])

        # Combine original data with normals and curvature
        if include_intensity:
            if include_reflectance:
                data_all = np.hstack((data_all[:, :3], normals, curvature[:, np.newaxis], data_all[:, -3:]))
            else:
                data_all = np.hstack((data_all[:, :3], normals, curvature[:, np.newaxis], data_all[:, -2:]))
        else:
            data_all = np.hstack((data_all[:, :3], normals, curvature[:, np.newaxis], data_all[:, -1][:, np.newaxis]))

        logging.info(f"Computed normals and curvature for the entire dataset. Data shape: {data_all.shape}")
    else:
        data_all = np.hstack((data_all[:, :3], normals, data_all[:, -1][:, np.newaxis]))
        logging.info(f"Computed normals for the entire dataset. Data shape: {data_all.shape}")

data_list = []
label_list = []
# Tile generation
for ind, point in enumerate(data_all):
    data_cur = point

    if compute_normals_flag:
        if compute_curvature_flag:
            if include_intensity:
                if include_reflectance:
                    if len(data_cur) == 10:  # Now using len(data_cur) instead of data_cur.shape[1]
                        data_list.append(data_cur[:9])  # Append data excluding label
                        label_list.append(data_cur[-1])  # Append corresponding labels
                        logging.info(f"Appended data with normals, curvature, intensity, and reflectance. Data shape: {data_cur[:9].shape}")
                elif len(data_cur) == 9:
                    data_list.append(data_cur[:8])  # Append data excluding label
                    label_list.append(data_cur[-1])  # Append corresponding labels
                    logging.info(f"Appended data with normals, curvature, and intensity. Data shape: {data_cur[:8].shape}")
            else:
                if len(data_cur) == 8:  # Expecting [x, y, z, nx, ny, nz, curvature, label]
                    data_list.append(data_cur[:7])  # Append data excluding label
                    label_list.append(data_cur[-1])  # Append corresponding labels
                    logging.info(f"Appended data with normals and curvature. Data shape: {data_cur[:7].shape}")
        else:
            if len(data_cur) == 7:  # Expecting [x, y, z, nx, ny, nz, label]
                data_list.append(data_cur[:6])  # Append data excluding label
                label_list.append(data_cur[-1])  # Append corresponding labels
                logging.info(f"Appended data with normals. Data shape: {data_cur[:6].shape}")
    else:
        if len(data_cur) == 4:  # Expecting [x, y, z, label]
            data_list.append(data_cur[:3])  # Append data excluding label
            label_list.append(data_cur[-1])  # Append corresponding labels
            logging.info(f"Appended data with coordinates only. Data shape: {data_cur[:3].shape}")
    # else:#todo recover
    #     data_list.append(data_cur[:, :3])
    # label_list.append(data_cur[:, -1])
    # if len(data_list) > 1:  #todo: Simplify the Loop
    #     break

print(f"Length of data_list: {len(data_list)}")
print(f"Length of label_list: {len(label_list)}")
# # Convert lists of tiles to a uniform 3D array for data and 2D array for labels
# data_array = np.array(data_list)
# label_array = np.array(label_list)

# Pre-allocate arrays based on the inclusion of normals, curvature, intensity, and reflectance
if compute_normals_flag:
    if compute_curvature_flag:
        if include_intensity:
            if include_reflectance:
                data_array = np.zeros((len(data_list), min_points, 9), dtype=np.float64)
            else:
                data_array = np.zeros((len(data_list), min_points, 8), dtype=np.float64)
        else:
            data_array = np.zeros((len(data_list), min_points, 7), dtype=np.float64)
    else:
        data_array = np.zeros((len(data_list), min_points, 6), dtype=np.float64)
else:
    data_array = np.zeros((len(data_list), min_points, 3), dtype=np.float64)

label_array = np.zeros((len(label_list), min_points), dtype=np.int64)

# Loop through each point and store in pre-allocated arrays
for i, data in enumerate(data_list):
    data_array[i, 0, :] = data  # Since min_points=1, we store each point directly at index 0
    label_array[i, 0] = label_list[i]  # Store the label similarly

# Save the arrays to an HDF5 file
with h5py.File(data_path_out, 'w') as hdf:
    hdf.create_dataset('data', data=data_array, dtype='float64')
    hdf.create_dataset('label', data=label_array, dtype='int32')

print(f'Saved {len(data_list)} tiles to {data_path_out}')
