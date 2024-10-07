import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Set up paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_file = os.path.join(BASE_DIR, 'scene_YellowArea_merged_2cm.csv')
data_file = os.path.join(BASE_DIR, 'scene_open_space_merged_2cm.csv')
# output_image_file = os.path.join(BASE_DIR, 'k_distance_graph_open_space_merged_2cm_removal.png')
output_image_file = os.path.join(BASE_DIR, 'k_distance_graph_open_space_merged_2cm_removal_.png')

# Load data
data_all = np.loadtxt(data_file, delimiter=',', skiprows=1)
point_cloud = data_all[:, :3]  # Extract the 3D coordinates

def plot_k_distance_graph(point_cloud, k=4, save_path='k_distance_graph.png'):
    """Plot the k-distance graph for DBSCAN parameter tuning and find the elbow point."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    # Extract the distances to the k-th nearest neighbor
    k_distances = distances[:, k-1]

    # Sort distances
    sorted_k_distances = np.sort(k_distances, axis=0)

    # Remove outliers
    filtered_k_distances = remove_outliers(sorted_k_distances)

    # Plot the sorted k-th nearest neighbor distances
    plt.plot(filtered_k_distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title('K-Distance Graph for DBSCAN')
    plt.grid(True)

    # Find the elbow point
    elbow_point = find_elbow_point(filtered_k_distances)
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    plt.text(elbow_point, filtered_k_distances[elbow_point],
             f'eps={filtered_k_distances[elbow_point]:.4f}', color='red')

    # Save the plot
    plt.savefig(save_path)
    plt.show()
    return filtered_k_distances[elbow_point]

def remove_outliers(distances):
    """Remove outliers using the IQR method."""
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return distances[(distances >= lower_bound) & (distances <= upper_bound)]

def find_elbow_point(sorted_distances):
    """Find the elbow point in the sorted k-distance graph."""
    num_points = len(sorted_distances)
    all_indices = np.arange(num_points)
    first_point = np.array([0, sorted_distances[0]])
    last_point = np.array([num_points - 1, sorted_distances[-1]])
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    distances_to_line = np.zeros(num_points)
    for i in range(num_points):
        point = np.array([all_indices[i], sorted_distances[i]])
        vec_from_first = point - first_point
        proj_length = np.dot(vec_from_first, line_vec_norm)
        proj_point = first_point + proj_length * line_vec_norm
        distances_to_line[i] = np.sqrt(np.sum((point - proj_point)**2))

    elbow_point = np.argmax(distances_to_line)
    return elbow_point

# Example usage
eps_value = plot_k_distance_graph(point_cloud, k=4, save_path=output_image_file)
print(f"Recommended eps value: {eps_value:.4f}")
