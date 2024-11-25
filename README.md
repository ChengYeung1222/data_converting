This repository contains tools for processing and converting point cloud datasets.

## Repository Contents

| File Name                              | Description                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|
| `data_split_normals_adjusted_buffer.py` | Splits and processes point cloud data into tiles, computes normals and curvature, and supports reflectance.   |
| `data_split_ts1.py`                   | Splits point cloud data for training and testing while handling normals, curvature, and intensity.            |
| `labelCount.py`                       | Counts and summarizes the distribution of labels.       |
| `normalVisualization_to_csv.py`       | Converts processed HDF5 point cloud data to CSV or TXT for visualization, with optional feature selection.      |
| `plot_k_distance_removal.py`          | Plots k-distance graphs to determine DBSCAN parameters and remove outliers. |

---

## Features

- **Point Cloud Tiling**:
  - `data_split_normals_adjusted_buffer.py`: Generate overlapping tiles with a buffer mechanism for regions of interest.
  - `data_split_ts1.py`: Split data into tiles of size 1, preparing it for support and query set generation.
- **Normals and Curvature Calculation**: Compute point normals and curvature for each point in the dataset, optionally adjusting normals based on detected planes.
- **Attributes Support**: Incorporate reflectance and intensity features during preprocessing.
- **Data and Filtering**: Tools for label counting, k-distance outlier removal, and label-based filtering.
- **Visualization Tools**: Convert point cloud data into CSV or TXT format for visualization.

---

## Prerequisites

- Python 3.6+
- Required Python packages:
  - `numpy`
  - `scipy`
  - `h5py`
  - `sklearn`
  - `matplotlib`
  - `pandas`

