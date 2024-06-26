import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = '/Users/paulgramlich/Developer/git/aiforgood/DATA/LBP/lbp_data_processed.csv'
df = pd.read_csv(file_path, index_col=0)

X = df.drop(columns=['gen12m'])
df = df.drop(columns=['recovered.12m'])
y_true = df['gen12m']
n_clusters = 8

'''X = df.drop(columns=['recovered.12m'])
df = df.drop(columns=['gen12m'])
y_true = df['recovered.12m']
n_clusters = 2'''


kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

nmi = normalized_mutual_info_score(y_true, y_pred)
ami = adjusted_mutual_info_score(y_true, y_pred)
def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)
purity = purity_score(y_true, y_pred)

silhouette_avg = silhouette_score(X, y_pred)
calinski_harabasz = calinski_harabasz_score(X, y_pred)
davies_bouldin = davies_bouldin_score(X, y_pred)

print(f"NMI: {nmi}")
print(f"AMI: {ami}")
print(f"Purity: {purity}")
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

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
