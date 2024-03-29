import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Function to load TIFF images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
    return images

# Load images from multiple folders
folder_paths = [
    "D:\\Yehmh\\test_py\\202301\\P00069\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00070\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00071\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00075\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00076\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00078\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00079\\5m_5m",
    "D:\\Yehmh\\test_py\\202301\\P00082\\5m_5m"
]

# Load images from all folders
images = []
for folder_path in folder_paths:
    images.extend(load_images_from_folder(folder_path))

# Convert images to feature vectors
data = np.array([image.flatten() for image in images])

# Define number of clusters (K)
num_clusters = 10  # Change this according to your requirement

# Apply K-means algorithm
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_

# Export coordinates and corresponding clusters to CSV
coordinates = np.array([(f"Image {i}", j) for i in range(len(images)) for j in range(images[i].shape[0]*images[i].shape[1])])
df = pd.DataFrame({"Image": coordinates[:, 0], "Coordinate": coordinates[:, 1], "Cluster": labels})
df.to_csv("coordinates_clusters.csv", index=False)
