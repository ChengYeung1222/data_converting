import numpy as np
import h5py
import os


def count_labels_csv(csv_file):
    """Function to count labels in a CSV file."""
    # Load the CSV file
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

    # Extract the label column (assuming it's the last column in the CSV)
    labels = data[:, -1]

    # Calculate unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Return label counts as a dictionary
    label_count_dict = dict(zip(unique_labels, counts))
    return label_count_dict


def count_labels_h5(h5_file):
    """Function to count labels in an HDF5 file."""
    with h5py.File(h5_file, 'r') as hdf:
        labels = hdf['label'][:]

    # Calculate unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Return label counts as a dictionary
    label_count_dict = dict(zip(unique_labels, counts))
    return label_count_dict


# Example usage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm.csv')
h5_file = os.path.join(BASE_DIR, 'scene_YellowArea_2cm_val_2048_fps_05m_normal_threshold200000.h5')

# Count labels from CSV
csv_label_counts = count_labels_csv(csv_file)
print("CSV Label Counts:", csv_label_counts)

# # Count labels from H5
# h5_label_counts = count_labels_h5(h5_file)
# print("H5 Label Counts:", h5_label_counts)




# 打开HDF5文件
# with h5py.File(hdf5_file_path, 'r') as file:
#     # 读取labels数据集
#     labels = file['label'][:]  # 假设标签存储在名为'label'的数据集中
#
#     # 计算每个标签的出现次数
#     unique_labels, counts = np.unique(labels, return_counts=True)
#
#     # 输出每个标签及其计数
#     print("Unique labels and their counts:")
#     for label, count in zip(unique_labels, counts):
#         print(f"Label: {label}, Count: {count}")


