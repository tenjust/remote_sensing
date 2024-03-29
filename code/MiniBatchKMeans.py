import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pandas as pd

# Function to recursively search for TIFF images in a folder
def find_tiff_images(folders):
    tiff_images = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".tif") or file.endswith(".tiff"):
                    tiff_images.append(os.path.join(root, file))
    return tiff_images

# Function to load and process images in batches
def process_images_in_batches(image_paths, batch_size, cluster_num):
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size
    print(num_batches)

    all_coordinates = []
    all_labels = []

    for i in range(num_batches):
        print(i, " batch")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)
        batch_paths = image_paths[start_idx:end_idx]

        batch_images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in batch_paths]
        batch_data = np.array([image.flatten() for image in batch_images])

        # Apply MiniBatchKMeans algorithm
        kmeans = MiniBatchKMeans(n_clusters=cluster_num)
        kmeans.fit(batch_data)

        # Get cluster labels
        labels = kmeans.labels_

        # Get coordinates
        coordinates = [(os.path.basename(path), j) for j in range(len(batch_images)) for path in batch_paths]

        all_coordinates.extend(coordinates)
        all_labels.extend(labels)

    return all_coordinates, all_labels

# Define folders containing TIFF images
folders = [
    "D:\\Yehmh\\test_py\\202301\\P00069\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00070\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00071\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00075\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00076\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00078\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00079\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00082\\5m_5m"
]

# Find TIFF images
tiff_image_paths = find_tiff_images(folders)

# Define batch size and number of clusters
batch_size = 2000  # Adjust as needed
cluster_num = 10   # Adjust as needed

# Process images in batches
coordinates, labels = process_images_in_batches(tiff_image_paths, batch_size, cluster_num)

# Create dataframe from coordinates and labels
df = pd.DataFrame({"Image": [coordinate[0] for coordinate in coordinates],
                   "Coordinate": [coordinate[1] for coordinate in coordinates],
                   "Cluster": labels})

# Export dataframe to CSV
df.to_csv("coordinates_clusters.csv", index=False)
