import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

file_path = '/Users/paulgramlich/Developer/git/aiforgood_patientclustering/DATA/LBP/lbp_data.csv'
data = pd.read_csv(file_path)

data.fillna(0, inplace=True)
data.replace('', 0, inplace=True)

'''data['gen12m'].fillna(0, inplace=True)
data['gen12m'].replace('', 0, inplace=True)

data.loc[:, data.columns != 'gen12m'] = data.loc[:, data.columns != 'gen12m'].fillna(-1)
data.loc[:, data.columns != 'gen12m'] = data.loc[:, data.columns != 'gen12m'].replace('', -1)'''

# Add all mappings/encodings

mapping_okom = {
    'Very likely': 11,
    'not at all likely': 1
}

data['okom0_full'] = data['okom0_full'].replace(mapping_okom)

mapping_gen12 = {
    'Completely recovered': 7,
    'Much improved': 6,
    'Slightly improved': 5,
    'Not changed': 4,
    'Slightly worsened': 3,
    'Much worsened': 2,
    'Worse than ever': 1
}

data['gen12m'] = data['gen12m'].replace(mapping_gen12)

categorical_columns = data.select_dtypes(include=['object']).columns

# Convert mixed-type columns to strings, except for '-1'
for column in categorical_columns:
    data[column] = data[column].apply(lambda x: str(x) if x != -1 else x)

label_encoder = LabelEncoder()
label_encodings = {
    'okom0_full': mapping_okom,
    'gen12': mapping_gen12
}

# Apply label encoding to each categorical column without the -1 values
for column in categorical_columns:
    mask = data[column] != -1
    data.loc[mask, column] = label_encoder.fit_transform(data.loc[mask, column])
    label_encodings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    data[column] = data[column].astype(int)

# Save the label encoding mappings to a text file
with open('/Users/paulgramlich/Developer/git/aiforgood_patientclustering/DATA/LBP/label_encodings.txt', 'w') as f:
    for column, mapping in label_encodings.items():
        f.write(f"Column: {column}\n")
        for original, encoded in mapping.items():
            f.write(f"  {original} -> {encoded}\n")
        f.write("\n")

print(data.head())
print("Categorical columns:", list(categorical_columns))

scaler = MinMaxScaler()

for column in data.columns:
    data[column] = data[column].astype(float)

first_column = data.columns[0]
index_data = data[first_column]

# Exclude 'gen12m' from scaling and start mapping from 1
data['gen12m'] = data['gen12m'].apply(lambda x: x + 1 if x > 0 else 0)

data['gen12m'] = data['gen12m'].astype(int)
data['recovered.12m'] = data['recovered.12m'].astype(int)

'''# Apply MinMaxScaler to all columns except 'gen12m'
for column in data.columns[1:]:
    if column != 'gen12m':
        if column != 'recovered.12m':
            mask = data[column] != -1
            if mask.sum() > 0:  # Ensure there are values to scale
                data_to_scale = data.loc[mask, column].values.reshape(-1, 1)
                data.loc[mask, column] = scaler.fit_transform(data_to_scale).flatten()'''

data[first_column] = index_data

output_file_path = '../LBP/lbp_data_processed_unscaled2.csv'
data.set_index(first_column, inplace=True)
data.to_csv(output_file_path)