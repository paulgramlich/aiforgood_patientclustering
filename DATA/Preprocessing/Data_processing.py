import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the processed dataset
processed_file_path = '../LBP/lbp_data_processed.csv'
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

# print(f"data_train_reshaped: {data_train_reshaped[0]}")
# print(f"labels_train shape: {labels_train.shape}")

X_train, X_test, y_train, y_test = train_test_split(data_train_reshaped, labels_train, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

mean_values = np.mean(data_train_reshaped, axis=0)

plt.imshow(mean_values, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Mean Values of Input Features in a 28x28 Image')
plt.show()

# Combine the data and labels for easier manipulation
data_with_labels = np.hstack((labels_train.reshape(-1, 1), data_train_padded))

# Calculate mean feature values for each label
mean_values_per_label = {}
for label in np.unique(labels_train):
    mean_values_per_label[label] = np.mean(data_with_labels[data_with_labels[:, 0] == label][:, 1:], axis=0)

# Convert to a DataFrame for easier plotting
mean_values_df = pd.DataFrame(mean_values_per_label).T

# Plot the mean values
plt.figure(figsize=(12, 8))
for i in range(mean_values_df.shape[1]):
    plt.plot(mean_values_df.index, mean_values_df[i], label=f'Feature {i}')

plt.xlabel('Labels')
plt.ylabel('Mean Feature Value')
plt.title('Mean Feature Values by Label')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()

labels = np.unique(labels_train)

for label in labels:
    mean_values_per_label[label] = np.mean(data_train_reshaped[labels_train == label], axis=0)

# Plot the mean values as heatmaps
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for i, label in enumerate(labels):
    axes[i].imshow(mean_values_per_label[label], cmap='viridis', aspect='auto')
    axes[i].set_title(f'Label {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()