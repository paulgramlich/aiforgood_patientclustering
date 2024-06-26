import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np

# Load the dataset
file_path = '/Users/paulgramlich/PycharmProjects/AlforGood_vfinal/dpsom/data/csv/lbp_data_processed.csv'
df = pd.read_csv(file_path, index_col=0)

# Separate features and labels
X = df.drop(columns=['gen12m'])
y_true = df['gen12m']

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)  # Adjust the number of components as needed
X_pca = pca.fit_transform(X)

# Perform k-means clustering with k-means++ initialization
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X_pca)

# Calculate NMI and AMI
nmi = normalized_mutual_info_score(y_true, y_pred)
ami = adjusted_mutual_info_score(y_true, y_pred)

# Calculate purity
def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

purity = purity_score(y_true, y_pred)

# Print the results
print(f"NMI: {nmi}")
print(f"AMI: {ami}")
print(f"Purity: {purity}")

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Elbow method for determining the optimal number of clusters
sse = []
silhouette_scores = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')

plt.show()