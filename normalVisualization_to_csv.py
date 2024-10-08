import numpy as np
import h5py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='visualization.log',
                    filemode='w')  # Use 'a' for append mode

INCLUDE_CURVATURE = True
INCLUDE_FLATNESS = False
INCLUDE_DENSITY = False
INCLUDE_INTENSITY = True
INCLUDE_REFLECTANCE = True


def merge_tiles_to_csv(h5_file_path, csv_file_path):
    # Load the data from the HDF5 file
    with h5py.File(h5_file_path, 'r') as hdf:
        data = hdf['data'][:]
        labels = hdf['label'][:]

    # Combine data and labels into one array
    num_points = data.shape[1]
    data = data.reshape(-1, 6)  # Flatten the data array to (N, 6)
    labels = labels.reshape(-1, 1)  # Flatten the label array to (N, 1)

    combined_data = np.hstack((data, labels))  # Combine data and labels into (N, 7)

    # Create a DataFrame
    columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'classification']
    df = pd.DataFrame(combined_data, columns=columns)

    # Export to CSV
    df.to_csv(csv_file_path, index=False)


def merge_tiles_to_txt_normals(h5_file_path, txt_file_path):
    # Load the data from the HDF5 file
    with h5py.File(h5_file_path, 'r') as hdf:
        data = hdf['data'][:]
        labels = hdf['label'][:]

    # Combine data and labels into one array
    num_points = data.shape[1]
    data = data.reshape(-1, 6)  # Flatten the data array to (N, 6)
    labels = labels.reshape(-1, 1)  # Flatten the label array to (N, 1)

    combined_data = np.hstack((data, labels))  # Combine data and labels into (N, 7)

    # Write to text file
    with open(txt_file_path, 'w') as txt_file:
        # Write header
        txt_file.write('x y z nx ny nz classification\n')
        # Write data
        np.savetxt(txt_file, combined_data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d')


def merge_tiles_to_txt(h5_file_path, txt_file_path, include_curvature=False, include_flatness=False,
                       include_density=False, include_intensity=False, include_reflectance=False):
    """
    Merge tiles and save them to a text file for visualization, with optional inclusion of curvature, flatness, and density.

    Args:
        h5_file_path: Path to the input HDF5 file.
        txt_file_path: Path to the output text file.
        include_curvature: Whether to include curvature in the output.
        include_flatness: Whether to include flatness in the output.
        include_density: Whether to include density in the output.
    """
    # Load the data from the HDF5 file
    with h5py.File(h5_file_path, 'r') as hdf:
        data = hdf['data'][:]
        labels = hdf['label'][:]

    # Combine data and labels into one array
    num_points = data.shape[1]

    # Determine the number of features based on the flags
    base_features = 6  # [x, y, z, nx, ny, nz]
    num_features = base_features
    if include_curvature:
        num_features += 1  # Add curvature
    if include_flatness:
        num_features += 1  # Add flatness
    if include_density:
        num_features += 1  # Add density
    if include_intensity:
        num_features += 1  # Add intensity
    if include_reflectance:
        num_features += 1

    # Reshape the data based on the selected features
    if 'ts1' in txt_file_path:
        pass
    else:
        data = data.reshape(-1, num_features)  # Flatten the data array to (N, num_features)
        labels = labels.reshape(-1, 1)  # Flatten the label array to (N, 1)

    if np.isnan(data).any():
        logging.warning("NaN values found.")
        print("NaN values found.")
    # if include_intensity:
    #     intensity_column_index = base_features  # The intensity column would be after the base features
    #     if include_curvature:
    #         intensity_column_index += 1
    #     if include_flatness:
    #         intensity_column_index += 1
    #     if include_density:
    #         intensity_column_index += 1
    #     if np.isnan(data[:, intensity_column_index]).any():
    #         logging.warning("NaN values found in the intensity column.")
    #
    #         # Remove rows where intensity has NaN values
    #         data = data[~np.isnan(data[:, intensity_column_index])]
    #         labels = labels[~np.isnan(data[:, intensity_column_index])]
    #         logging.info(f"Filtered NaN values from intensity. New data shape: {data.shape}")

    # Combine data and labels into a single array
    if 'ts1' in txt_file_path:
        data_squeezed = np.squeeze(data, axis=1)  # 去掉第二个维度，结果形状为 (4828486, 9)
        combined_data = np.hstack((data_squeezed, labels))  # 合并数据和标签
    else:
        combined_data = np.hstack((data, labels))  # Combine data and labels

    # Write to text file
    with open(txt_file_path, 'w') as txt_file:
        # Write the header based on the flags
        header = 'x y z nx ny nz'
        if include_curvature:
            header += ' curvature'
        if include_flatness:
            header += ' flatness'
        if include_density:
            header += ' density'
        if include_intensity:
            header += ' intensity'
        if include_reflectance:
            header += ' reflectance'
        header += ' classification\n'
        txt_file.write(header)

        # Determine the format string based on the number of features
        fmt_string = '%.6f ' * num_features + '%d\n'  # Format for the features and classification label

        # Write the combined data to the text file
        np.savetxt(txt_file, combined_data, fmt=fmt_string.strip())


def merge_tiles_to_csv_expired(h5_file_path, csv_file_path, relevant_tiles):
    # Load the data from the HDF5 file
    with h5py.File(h5_file_path, 'r') as hdf:
        data = hdf['data'][:]
        labels = hdf['label'][:]

    # Combine data and labels into one array
    num_points = data.shape[1]
    data = data.reshape(-1, 6)  # Flatten the data array to (N, 6)
    labels = labels.reshape(-1, 1)  # Flatten the label array to (N, 1)

    combined_data = np.hstack((data, labels))  # Combine data and labels into (N, 7)

    # Create a DataFrame
    columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'classification']
    df = pd.DataFrame(combined_data, columns=columns)

    # Filter the DataFrame to include only relevant tiles
    tile_indices = []
    for i, k in relevant_tiles:
        # Compute the index range for the tile (i, k)
        start_idx = (i * len(x_lim) + k) * num_points
        end_idx = start_idx + num_points
        tile_indices.extend(range(start_idx, end_idx))

    df_filtered = df.iloc[tile_indices]

    # Export to CSV
    df_filtered.to_csv(csv_file_path, index=False)


# Specify the paths to your HDF5 file and the output CSV file
# h5_file_path = 'scene_open_space_merged_2cm_train_2048_fps_05m_normal_41tiles.h5'
# csv_file_path = 'scene_open_space_merged_2cm_train_2048_fps_05m_normal_41tiles.csv'
# h5_file_path = 'scene_YellowArea_2cm_val_2048_fps_05m_normal_adjusted.h5'
# csv_file_path = 'scene_YellowArea_2cm_val_2048_fps_05m_normal_adjusted.csv'
# txt_file_path = './scene_YellowArea_2cm_val_2048_fps_05m_normal_adjusted.txt'
# csv_file_path = './scene_open_space_merged_2cm_train_2048_fps_05m_normal_adjusted_41tiles.csv'
# h5_file_path = './scene_open_space_merged_2cm_train_2048_fps_05m_normal_adjusted_41tiles_notadjusted.h5'
# txt_file_path = './scene_open_space_merged_2cm_train_2048_fps_05m_normal_adjusted_41tiles_notadjusted.txt'
# h5_file_path = './scene_open_space_merged_1cm_train_2048_fps_05m_normal_adjusted_41tiles_entire.h5'
# txt_file_path = './scene_open_space_merged_1cm_train_2048_fps_05m_normal_adjusted_41tiles_entire.txt'
# h5_file_path = './scene_YellowArea_2cm_val_2048_fps_05m_normal_curvature.h5'
# txt_file_path = './scene_YellowArea_2cm_val_2048_fps_05m_normal_curvature.txt'
# h5_file_path = './scene_open_space_merged_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter.h5'
# txt_file_path = './scene_open_space_merged_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter.txt'
# h5_file_path = './scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined.h5'
# txt_file_path = './scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined.txt'
# h5_file_path = './scene_YellowArea_2cm_val_2048_fps_05m_normal_curvature_flatness_41tiles.h5'
# txt_file_path = './scene_YellowArea_2cm_val_2048_fps_05m_normal_curvature_flatness_41tiles.txt'
# h5_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature.h5'
# txt_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature.txt'
# h5_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature_intensity.h5'
# txt_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature_intensity.txt'
# h5_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_withfilter_reflectance.h5'
# txt_file_path = './scene_RedArea_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_withfilter_reflectance.txt'
h5_file_path = './scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined_ts1.h5'
txt_file_path = './scene_open_space_2cm_train_2048_fps_2m_normal_curvature_intensity_buffer_wofilter_reflectance_refined_ts1.txt'

# relevant_tiles = [(i, k) for i in range(1) for k in range(len(x_lim))]

# merge_tiles_to_csv(h5_file_path, csv_file_path)
merge_tiles_to_txt(h5_file_path, txt_file_path, include_curvature=INCLUDE_CURVATURE, include_flatness=INCLUDE_FLATNESS,
                   include_density=INCLUDE_DENSITY, include_intensity=INCLUDE_INTENSITY,
                   include_reflectance=INCLUDE_REFLECTANCE)

# print(f'Saved merged tiles to {csv_file_path}')
print(f'Saved merged tiles to {txt_file_path}')
