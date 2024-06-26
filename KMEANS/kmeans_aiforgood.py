import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np

# Load the dataset
file_path = '/Users/paulgramlich/Developer/git/aiforgood/DATA/LBP/lbp_data_processed.csv'
df = pd.read_csv(file_path, index_col=0)

# Separate features and labels
X = df.drop(columns=['gen12m'])
df = df.drop(columns=['recovered.12m'])
y_true = df['gen12m']
n_clusters = 8

'''X = df.drop(columns=['recovered.12m'])
df = df.drop(columns=['gen12m'])
y_true = df['recovered.12m']
n_clusters = 2'''


kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred = kmeans.fit_predict(X)

# Calculate NMI and AMI
nmi = normalized_mutual_info_score(y_true, y_pred)
ami = adjusted_mutual_info_score(y_true, y_pred)

# Calculate purity
def purity_score(y_true, y_pred):
    # Compute contingency matrix (also called confusion matrix)
    contingency_matrix = pd.crosstab(y_true, y_pred)
    # Return purity
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

purity = purity_score(y_true, y_pred)

# Print the results
print(f"NMI: {nmi}")
print(f"AMI: {ami}")
print(f"Purity: {purity}")

