import pandas as pd

# 1. Load your CSV file
df = pd.read_csv('attributes_00.csv')

# 2. Create a new column 'label' based on the text in the file_name
# If it contains 'anomaly', assign it a 1 (Faulty). Otherwise, assign it a 0 (Healthy).
df['label'] = df['file_name'].apply(lambda x: 1 if 'anomaly' in x else 0)

# 3. Create a readable status column just for your own visual check
df['status'] = df['file_name'].apply(lambda x: 'Faulty' if 'anomaly' in x else 'Healthy')

# 4. Check how many of each you have
print(df['status'].value_counts())