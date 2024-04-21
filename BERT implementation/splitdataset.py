import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset using pandas
dataset_path = "C:/Users/huseinkantarci/Desktop/keyword-transformer/bert-cased/preprocessed_no_numbers.csv"
df = pd.read_csv(dataset_path)
df = df.dropna(subset=['text', 'keywords'])


# Split the dataset into train, validation, and test sets
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Reset the indices of the dataframes
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Save the datasets to CSV files
train_data.to_csv("C:/Users/huseinkantarci/Desktop/keyword-transformer/bert-cased/split_dataset/train.csv", index=False)
val_data.to_csv("C:/Users/huseinkantarci/Desktop/keyword-transformer/bert-cased/split_dataset/val.csv", index=False)
test_data.to_csv("C:/Users/huseinkantarci/Desktop/keyword-transformer/bert-cased/split_dataset/test.csv", index=False)