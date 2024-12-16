import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
csv_file = "D:\\archive\\im2latex_train.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Use only 25% of the original data
subset_data = data.sample(frac=0.05, random_state=42)  # Randomly sample 5% of the data

# Calculate the sizes of the splits for train, validation, and test
train_frac = 0.60
val_frac = 0.25
test_frac = 0.15

# Step 1: Split into train and temp (validation + test)
train_data, temp_data = train_test_split(subset_data, test_size=(1 - train_frac), random_state=42)

# Step 2: Split temp into validation and test
val_data, test_data = train_test_split(temp_data, test_size=(test_frac / (test_frac + val_frac)), random_state=42)

train_data.to_csv("train_split.csv", index=False)
val_data.to_csv("val_split.csv", index=False)
test_data.to_csv("test_split.csv", index=False)