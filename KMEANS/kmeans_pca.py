import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/paulgramlich/PycharmProjects/AlforGood_vfinal/dpsom/data/csv/lbp_data_processed.csv'
df = pd.read_csv(file_path, index_col=0)

# Separate features and labels
X = df.drop(columns=['gen12m'])
df = df.drop(columns=['recovered.12m'])
y_true = df['gen12m']

# Apply PCA for dimensionality reduction to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform k-means clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

# Plotting the true labels
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50)
plt.title('True Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()

# Plotting the predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=50)
plt.title('Predicted Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()

plt.show()