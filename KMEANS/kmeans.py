import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = '../DATA/LBP/lbp_data_processed.csv'
df = pd.read_csv(file_path, index_col=0)

X = df.drop(columns=['gen12m'])
X = X.drop(columns=['recovered.12m'])
y_true = df['gen12m']
n_clusters = 8

'''X = df.drop(columns=['recovered.12m'])
df = df.drop(columns=['gen12m'])
y_true = df['recovered.12m']
n_clusters = 2'''

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.5, random_state=42)

n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

nmi = normalized_mutual_info_score(y_test, y_pred)
ami = adjusted_mutual_info_score(y_test, y_pred)
def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)
purity = purity_score(y_test, y_pred)

silhouette_avg = silhouette_score(X_test, y_pred)
calinski_harabasz = calinski_harabasz_score(X_test, y_pred)
davies_bouldin = davies_bouldin_score(X_test, y_pred)

print(f"NMI: {nmi}")
print(f"AMI: {ami}")
print(f"Purity: {purity}")
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', s=50)
plt.title('True Labels')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis', s=50)
plt.title('Predicted Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()

plt.show()