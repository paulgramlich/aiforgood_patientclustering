import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the processed dataset
processed_file_path = '../LBP/lbp_data_processed.csv'
processed_data = pd.read_csv(processed_file_path, index_col=0)

labels_train = processed_data.iloc[:, 0].values
data_train = processed_data.iloc[:, 1:].values

print(processed_data)

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

plt.imshow(mean_values, cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
plt.title('Mean Values of Input Features in a 28x28 Image')
plt.show()

data_with_labels = np.hstack((labels_train.reshape(-1, 1), data_train_padded))

mean_values_per_label = {}
for label in np.unique(labels_train):
    mean_values_per_label[label] = np.mean(data_with_labels[data_with_labels[:, 0] == label][:, 1:], axis=0)

mean_values_df = pd.DataFrame(mean_values_per_label).T

plt.figure(figsize=(12, 8))
for i in range(mean_values_df.shape[1]):
    plt.plot(mean_values_df.index, mean_values_df[i], label=f'Feature {i}')

plt.xlabel('Labels')
plt.ylabel('Mean Feature Value')
plt.title('Mean Feature Values by Label')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()

labels = np.arange(8)

for label in labels:
    if np.any(labels_train == label):
        mean_values_per_label[label] = np.mean(data_train_reshaped[labels_train == label], axis=0)
    else:
        mean_values_per_label[label] = np.zeros((28, 28))

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for i, label in enumerate(labels):
    im = axes[i].imshow(mean_values_per_label[label], cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[i].set_title(f'Label {label}')
    axes[i].axis('off')

fig.subplots_adjust(bottom=0.1)
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()