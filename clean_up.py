import pandas as pd
import os

# Load the dataset
script_dir = os.path.dirname(__file__)
cleaned_file = os.path.join(script_dir, 'cleaned.csv')  # Output file path
file_path = os.path.join(script_dir, 'cleaned_dataset.csv')  # Output file path
df = pd.read_csv(file_path)

# Convert 'label' to integer and handle null values
df['label'] = df['label'].fillna(-1).astype(int)
df = df[df['label'].isin([0, 1])]

# Check the counts for each label after conversion
count_label_0 = df[df['label'] == 0].shape[0]
count_label_1 = df[df['label'] == 1].shape[0]

print("Count of label 0:", count_label_0)
print("Count of label 1:", count_label_1)

# Adjust the number of samples based on availability
num_samples_0 = 50
# min(50, count_label_0)
num_samples_1 = 50
# min(50, count_label_1)

selected_zeros = df[df['label'] == 0].sample(num_samples_0, random_state=42)
selected_ones = df[df['label'] == 1].sample(num_samples_1, random_state=42)

# Combine the selected rows
combined_selected_rows = pd.concat([selected_zeros, selected_ones])

# Save the new dataset to a CSV file
combined_selected_rows.to_csv(cleaned_file, index=False)

print("New dataset saved to:", cleaned_file)
