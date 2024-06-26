import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the processed dataset
processed_file_path = '/Users/paulgramlich/Developer/SOM-VAE_aiforgood/SOM-VAE/data/LBP/lbp_data_processed.csv'
processed_data = pd.read_csv(processed_file_path, index_col=0)

labels_train = processed_data.iloc[:, 0].values
data_train = processed_data.iloc[:, 1:].values

print(processed_data)# .iloc[:, 0]

num_samples = data_train.shape[0]
num_features = data_train.shape[1]
target_num_features = 28 * 28

if num_features < target_num_features:
    data_train_padded = np.pad(data_train, ((0, 0), (0, target_num_features - num_features)), 'constant', constant_values=0)
else:
    data_train_padded = data_train

data_train_reshaped = np.reshape(data_train_padded, (num_samples, 28, 28, 1))

print(f"data_train_reshaped: {data_train_reshaped[0]}")
print(f"labels_train shape: {labels_train.shape}")

X_train, X_test, y_train, y_test = train_test_split(data_train_reshaped, labels_train, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")