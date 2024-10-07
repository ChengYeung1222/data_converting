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
mode = 'train'
# mode = 'val'  # todo regenerate train_adjusted 0124
compute_normals_flag = True
compute_curvature_flag = True
normals_adjusted = True
filter_small_categories = False  # todo
include_intensity = True
include_reflectance = True
min_points = 2048  # todo
std = 0.35
tile_size = 2  # todo: 5, .5
overlap = 0.8
dynamic_overlap = True
initial_overlap = 0.8
m = 1
d = tile_size - (tile_size * overlap)

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_processing_split_open_2m_buffer_wofilter_reflectance_refined.log',
                    filemode='w')  # Use 'a' for append mode

# Example of logging
logging.info("Started the script.")

# Set up paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_5cm.csv')
# data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_1cm.csv') #todo
data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm_refined.txt')
# data_file = os.path.join(BASE_DIR, 'scene_YellowArea_merged_2cm.txt')
# data_file = os.path.join(BASE_DIR, 'scene_RedArea_merged_2cm_refined.txt')
# train_data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_1cm.csv')
train_data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm.csv')
# CSV Label Counts: {0.0: 422197, 1.0: 1157999, 2.0: 1217007, 3.0: 137219, 4.0: 700834, 5.0: 5539, 6.0: 184051, 7.0: 321450, 8.0: 45388, 9.0: 7073, 10.0: 38276, 11.0: 11187, 12.0: 1227, 13.0: 423, 14.0: 4594, 15.0: 1238, 16.0: 1147}
# data_path_out = os.path.join(BASE_DIR, 'scene_open_space_merged_5cm_train_2048_fps_2m.h5')
# data_path_out = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm_train_2048_fps_05m_normal.h5')
# data_path_out = os.path.join(BASE_DIR, 'scene_YellowArea_2cm_val_2048_fps_05m_normal_adjusted.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_open_space_merged_2cm_train_2048_fps_05m_normal_adjusted_41tiles_entire.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_open_space_merged_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_YellowArea_2cm_val_2048_fps_05m_normal_adjusted_entire_200.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_YellowArea_2cm_val_2048_fps_2m_normal_curvature_intensity.h5')
# data_path_out = os.path.join(BASE_DIR,
#                              'scene_YellowArea_2cm_val_2048_fps_2m_normal_curvature_intensity_buffer_wofilter.h5')
data_path_out = os.path.join(BASE_DIR,
                             'scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined.h5')
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

min_count = 130000
if filter_small_categories:
    # if mode == 'val':
    if 'open_space' not in data_file:
        train_data_all = np.loadtxt(train_data_file, delimiter=',', skiprows=1)
        train_data_all = np.hstack((train_data_all[:, :3], train_data_all[:, -1][:, np.newaxis]))
        train_labels = train_data_all[:, -1]
    labels = data_all[:, -1]
    # print("Data shape before filtering:", data_all.shape)
    logging.info(f"Data shape before filtering: {data_all.shape}")
    # if mode == 'val':
    if 'open_space' not in data_file:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
    else:
        unique_labels, counts = np.unique(labels, return_counts=True)
    filtered_labels = unique_labels[counts >= min_count]
    # filtered_labels = [0, 1, 2, 3, 4, 6, 7, 8]
    logging.info(f"Filtered labels: {filtered_labels}")
    mask = np.isin(labels, filtered_labels)
    data_all = data_all[mask]
    # print("Filtered data shape:", data_all.shape)
    logging.info(f"Filtered data shape: {data_all.shape}")

# Coordinate limits
x_min, x_max = data_all[:, 0].min(), data_all[:, 0].max()
y_min, y_max = data_all[:, 1].min(), data_all[:, 1].max()

x_lim = np.arange(x_min - m, (x_max - tile_size) + m, d)
y_lim = np.arange(y_min - m, (y_max - tile_size) + m, d)


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


#
# def adjust_normals_ground_ceiling(point_cloud, normals, labels, major_planes):
#     """Adjust normals based on detected planes."""
#     adjusted_normals = normals.copy()
#
#     for label, normal in major_planes:
#         points_idx = np.where(labels == label)[0]
#         for i in points_idx:
#             if is_ground_plane(normal):
#                 if normals[i][2] < 0:  # Adjust if normal is not pointing up
#                     adjusted_normals[i] = -normals[i]
#             elif is_ceiling_plane(normal):
#                 if normals[i][2] > 0:  # Adjust if normal is not pointing down
#                     adjusted_normals[i] = -normals[i]
#
#     return adjusted_normals
#
#
# def is_ground_plane(normal):
#     return normal[2] > 0.8  # Example condition for ground plane
#
#
# def is_ceiling_plane(normal):
#     return normal[2] < -0.8  # Example condition for ceiling plane


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


def farthest_point_sampling_kdt(points, k, max_points):
    start_time = time.time()

    try:
        if not np.isfinite(points).all():
            points = points[np.isfinite(points).all(axis=1)]
            logging.info(f"Filtered non-finite points, new shape: {points.shape}")

        num_points = points.shape[0]

        if num_points < k:
            raise ValueError("Not enough points after filtering to perform sampling.")

        initial_idx = np.random.randint(num_points)
        farthest_pts = [initial_idx]
        remaining_pts = list(range(num_points))
        remaining_pts.remove(initial_idx)

        kdtree = KDTree(points[:, :3])
        logging.info("KDTree initialization successful.")

        dynamic_k = max(10, int(num_points * 0.01))
        for i in range(1, k):
            last_selected_point = points[farthest_pts[-1], :3]
            distances, indices = kdtree.query(last_selected_point, k=dynamic_k)
            valid_distances = {idx: dist for idx, dist in zip(indices, distances) if
                               idx not in farthest_pts and idx < num_points}

            if not valid_distances:
                raise ValueError("No valid distances found; all points may already be sampled.")

            new_point_idx = max(valid_distances, key=valid_distances.get)
            farthest_pts.append(new_point_idx)
            remaining_pts.remove(new_point_idx)

        if len(farthest_pts) < k:
            additional_pts = np.random.choice(remaining_pts, k - len(farthest_pts), replace=False)
            farthest_pts.extend(additional_pts)
            logging.warning(
                f"Used random sampling to fill the remaining points. Required: {k}, Found: {len(farthest_pts)}.")

        end_time = time.time()
        logging.info(f"Farthest Point Sampling completed in {end_time - start_time:.2f} seconds.")
        return points[farthest_pts]

    except Exception as e:
        logging.error(f"Error during sampling: {e}")
        return None
        # Optionally re-raise the error if you want it to propagate up:
        # raise


def farthest_point_sampling_kdt_complicated(points, k, max_points):
    start_time = time.time()  # Start timing
    num_points = points.shape[0]
    # if num_points > max_points:
    #     indices = np.random.choice(len(points), max_points, replace=False)
    #     points = points[indices]
    #     num_points = max_points
    # farthest_pts = np.zeros(k, dtype=np.int)
    # farthest_pts[0] = np.random.randint(len(points))

    initial_idx = np.random.randint(len(points))
    farthest_pts = [initial_idx]
    remaining_pts = list(range(num_points))
    remaining_pts.remove(initial_idx)
    # Data validation and preprocessing

    if not np.isfinite(points).all():
        points = points[np.isfinite(points).all(axis=1)]
        logging.info(f"Filtered non-finite points, new shape: {points.shape}")

    if points.shape[0] < k:
        logging.error("Not enough points after filtering to perform sampling.")
        return None

    try:
        logging.info(f"Initializing KDTree with data shape: {points[:, :3].shape}")
        kdtree = KDTree(points[:, :3])
        logging.info("KDTree initialization successful.")
    except Exception as e:
        logging.error(f"KDTree initialization failed: {e}")
        return None

    # distances = np.full(num_points, np.inf)
    dynamic_k = max(10, int(num_points * 0.01))
    distances = np.full(dynamic_k, np.inf)

    failure_count = 0
    for i in range(1, k):
        # Get the distances to the nearest previously selected point
        # new_distances, _ = kdtree.query(points[farthest_pts[i - 1]], k=num_points)
        last_selected_point = points[farthest_pts[-1], :3]
        distances, indices = kdtree.query(last_selected_point, k=dynamic_k)
        valid_distances = {idx: dist for idx, dist in zip(indices, distances) if idx not in farthest_pts}
        # if not valid_distances:
        #     logging.info("No more unique points to add; breaking out of loop.")
        #     break
        if not valid_distances:
            dynamic_k *= 2  # Double the search range
            failure_count += 1
            logging.info("No valid distances found; increasing search range and retrying.")

            if failure_count > 3:  # Too many failures, choose a new random start point
                new_start = np.random.choice(list(remaining_pts))
                farthest_pts.append(new_start)
                remaining_pts.remove(new_start)
                failure_count = 0  # Reset failure count
                logging.info(f"Resetting starting point to {new_start}")
            continue
        new_point_idx = max(valid_distances, key=valid_distances.get)
        farthest_pts.append(new_point_idx)
        remaining_pts.remove(new_point_idx)
        # new_distances, _ = kdtree.query(points[farthest_pts[i - 1]], k=dynamic_k)
        # distances = np.minimum(distances, new_distances)
        # farthest_pts[i] = np.argmax(distances)
    if len(farthest_pts) < k:
        logging.warning(
            f"FPS could not find enough points. Required: {k}, Found: {len(farthest_pts)}. Using random sampling to fill the remaining points.")
        additional_pts = np.random.choice(remaining_pts, k - len(farthest_pts), replace=False)
        farthest_pts.extend(additional_pts)
    end_time = time.time()  # End timing
    logging.info(f"Farthest Point Sampling completed in {end_time - start_time:.2f} seconds.")
    # Set the initial maximum distance
    # distances = np.zeros(num_points)
    # max_distances = np.full(num_points, np.inf)
    # for i in range(1, k):
    #     # Update distances for all points as the minimum of the existing max_distances or the distance to the new farthest point
    #     distances, _ = kdtree.query(points, k=1)
    #     max_distances = np.minimum(max_distances, distances)
    #
    #     # Select the new farthest point based on the maximum distance
    #     farthest_pts[i] = np.argmax(max_distances)

    return points[farthest_pts]


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
dropped_points_count = 0
# Buffer to hold points that are insufficient to form a full tile
buffer = []
# Tile generation
for i in y_lim:
    # if len(data_list) > 1:  # todo: Simplify the Loop
    #     break
    for k in x_lim:
        # logging.info(f"Start generating tile: {k}")
        ind = np.where((k <= data_all[:, 0]) & (data_all[:, 0] <= k + tile_size) &
                       (i <= data_all[:, 1]) & (data_all[:, 1] <= i + tile_size))
        data_cur = data_all[ind]
        if data_cur.shape[0] < min_points:
            if data_cur.shape[0] != 0:
                buffer.append(data_cur)
                combined_buffer = np.vstack(buffer)
                if combined_buffer.shape[0] >= min_points:
                    idx = np.random.choice(combined_buffer.shape[0], min_points, replace=False)
                    new_tile = combined_buffer[idx]
                    if include_intensity:
                        if include_reflectance:
                            if new_tile.shape[1] == 10:
                                data_list.append(new_tile[:, :9])  # Append data excluding label
                                label_list.append(new_tile[:, -1])  # Append corresponding labels
                                logging.info(
                                    f"Appended data with normals, curvature, intensity and reflectance (<). Data shape: {new_tile[:, :9].shape}")
                        elif new_tile.shape[1] == 9:
                            data_list.append(new_tile[:, :8])  # Append data excluding label
                            label_list.append(new_tile[:, -1])  # Append corresponding labels
                            logging.info(
                                f"Appended data with normals, curvature and intensity (<). Data shape: {new_tile[:, :8].shape}")
                    remaining_points = np.delete(combined_buffer, idx, axis=0)
                    buffer = [remaining_points] if remaining_points.shape[0] > 0 else []
                # logging.info(f"Skipping tile at ({i}, {k}) due to insufficient points: {data_cur.shape[0]}")
                # dropped_points_count += data_cur.shape[0]
                # continue  # Here, consider if there's a better approach than just skipping.

        if mode == 'train':
            h, _ = np.histogram(data_cur[:, -1])
            st = np.std(h / float(np.sum(h)), ddof=1)
            if st > std:
                continue

        # Subsample if there are more than min_points
        if data_cur.shape[0] > min_points:
            # if compute_normals_flag:
            #     if normals_adjusted:
            #         # normals = compute_normals_adjusted(data_cur[:, :3],labels=data_cur[:, -1])
            #         normals = compute_normals_adjusted(data_cur[:, :3])
            #     else:
            #         # Compute normals before subsampling
            #         normals = compute_normals(data_cur[:, :3])
            #     data_cur = np.hstack((data_cur[:, :3], normals, data_cur[:, -1][:, np.newaxis]))
            #     logging.info(f"Computed normals and updated data_cur shape: {data_cur.shape}")
            # if mode == 'train':
            # if data_cur.shape[0] > min_points * 1.5:
            #     data_cur = farthest_point_sampling_kdt(data_cur, min_points, max_points=10000)
            # else:
            #     idx = np.random.choice(data_cur.shape[0], min_points, replace=False)
            #     data_cur = data_cur[idx]
            if data_cur.shape[0] > min_points * 1.5:
                sampled_data = farthest_point_sampling_kdt(data_cur, min_points, max_points=10000)
                if sampled_data is None:
                    idx = np.random.choice(data_cur.shape[0], min_points, replace=False)
                    data_cur = data_cur[idx]
                else:
                    data_cur = sampled_data
            else:
                idx = np.random.choice(data_cur.shape[0], min_points, replace=False)
                data_cur = data_cur[idx]

            # Append data and labels conditionally, ensuring both lists are updated together
            if compute_normals_flag:
                if compute_curvature_flag:
                    if include_intensity:
                        if include_reflectance:
                            if data_cur.shape[1] == 10:
                                data_list.append(data_cur[:, :9])  # Append data excluding label
                                label_list.append(data_cur[:, -1])  # Append corresponding labels
                                logging.info(
                                    f"Appended data with normals, curvature, intensity and reflectance. Data shape: {data_cur[:, :9].shape}")
                        elif data_cur.shape[1] == 9:
                            data_list.append(data_cur[:, :8])  # Append data excluding label
                            label_list.append(data_cur[:, -1])  # Append corresponding labels
                            logging.info(
                                f"Appended data with normals, curvature and intensity. Data shape: {data_cur[:, :8].shape}")
                    else:
                        if data_cur.shape[1] == 8:  # Expecting [x, y, z, nx, ny, nz, curvature, label]
                            data_list.append(data_cur[:, :7])  # Append data excluding label
                            label_list.append(data_cur[:, -1])  # Append corresponding labels
                            logging.info(
                                f"Appended data with normals and curvature. Data shape: {data_cur[:, :7].shape}")
                    # else:
                    #     logging.error(f"Unexpected shape in data_cur: {data_cur.shape}")
                    #     continue  # Skip to next iteration if shape is incorrect
                else:
                    if data_cur.shape[1] == 7:  # Expecting [x, y, z, nx, ny, nz, label]
                        data_list.append(data_cur[:, :6])  # Append data excluding label
                        label_list.append(data_cur[:, -1])  # Append corresponding labels
                        logging.info(f"Appended data with normals. Data shape: {data_cur[:, :6].shape}")
                    else:
                        logging.error(f"Unexpected shape in data_cur: {data_cur.shape}")
                        continue  # Skip to next iteration if shape is incorrect
            else:
                if data_cur.shape[1] == 4:  # Expecting [x, y, z, label]
                    data_list.append(data_cur[:, :3])  # Append data excluding label
                    label_list.append(data_cur[:, -1])  # Append corresponding labels
                    logging.info(f"Appended data with coordinates only. Data shape: {data_cur[:, :3].shape}")
                else:
                    logging.error(f"Unexpected shape in data_cur: {data_cur.shape}")
                    continue  # Skip to next iteration if shape is incorrect
            # else:#todo recover
            #     data_list.append(data_cur[:, :3])
            # label_list.append(data_cur[:, -1])
            # if len(data_list) > 1:  #todo: Simplify the Loop
            #     break

            # Handle the remaining points
            remaining_idx = np.setdiff1d(np.arange(data_cur.shape[0]), idx)
            # Check if remaining_idx is not empty
            if remaining_idx.size > 1:
                remaining_points = data_cur[remaining_idx]
                # Further processing with remaining_points
                # Add remaining points to the buffer
                if remaining_points.shape[0] > 0:
                    buffer.append(remaining_points)
                    combined_buffer = np.vstack(buffer)
                    # Check if the buffer can form a full tile again after adding remaining points
                    if combined_buffer.shape[0] >= min_points:
                        idx = np.random.choice(combined_buffer.shape[0], min_points, replace=False)
                        new_tile = combined_buffer[idx]

                        # Add the new tile to the data list
                        if include_intensity:
                            if include_reflectance:
                                if new_tile.shape[1] == 10:
                                    data_list.append(new_tile[:, :9])  # Append data excluding label
                                    label_list.append(new_tile[:, -1])  # Append corresponding labels
                                    logging.info(
                                        f"Appended data with normals, curvature, intensity and reflectance (>). Data shape: {new_tile[:, :9].shape}")
                            elif new_tile.shape[1] == 9:
                                data_list.append(new_tile[:, :8])  # Append data excluding label
                                label_list.append(new_tile[:, -1])  # Append corresponding labels
                                logging.info(
                                    f"Appended data with normals, curvature and intensity (>). Data shape: {new_tile[:, :8].shape}")

                        # Remove the used points from the buffer
                        remaining_points = np.delete(combined_buffer, idx, axis=0)
                        buffer = [remaining_points] if remaining_points.shape[0] > 0 else []
            else:
                logging.info("No remaining points to process. Skipping this step.")

# Final check: If there are leftover points in the buffer, try merging with the next available tile
if buffer and len(buffer) > 0:
    combined_buffer = np.vstack(buffer)
    if combined_buffer.shape[0] >= min_points:
        idx = np.random.choice(combined_buffer.shape[0], min_points, replace=False)
        final_tile = combined_buffer[idx]
        if include_intensity:
            if include_reflectance:
                if final_tile.shape[1] == 10:
                    data_list.append(final_tile[:, :9])  # Append data excluding label
                    label_list.append(final_tile[:, -1])  # Append corresponding labels
                    logging.info(
                        f"Appended data with normals, curvature, intensity and reflectance (final). Data shape: {final_tile[:, :9].shape}")
            elif final_tile.shape[1] == 9:
                data_list.append(final_tile[:, :8])  # Append data excluding label
                label_list.append(final_tile[:, -1])  # Append corresponding labels
                logging.info(
                    f"Appended data with normals, curvature and intensity (final). Data shape: {final_tile[:, :8].shape}")

logging.info(f"Total number of dropped points due to insufficient points in tiles: {dropped_points_count}")
print(f"Length of data_list: {len(data_list)}")
print(f"Length of label_list: {len(label_list)}")
# # Convert lists of tiles to a uniform 3D array for data and 2D array for labels
# data_array = np.array(data_list)
# label_array = np.array(label_list)

# Convert lists of tiles to uniform arrays
# max_length = max(len(data) for data in data_list)
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

for i, data in enumerate(data_list):
    length = len(data)
    # if compute_normals_flag:
    #     if data.shape[1] == 6:
    #         data_array[i, :length, :] = data
    #         label_array[i, :length] = label_list[i]
    #     else:
    #         logging.error(f"Unexpected shape in data_list[{i}]: {data.shape}")
    # else:
    data_array[i, :length, :] = data
    label_array[i, :length] = label_list[i]

# Save to HDF5
with h5py.File(data_path_out, 'w') as hdf:
    hdf.create_dataset('data', data=data_array, dtype='float64')
    hdf.create_dataset('label', data=label_array, dtype='int32')

print(f'Saved {len(data_list)} tiles to {data_path_out}')
