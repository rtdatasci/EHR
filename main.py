# main.py

from sklearn.datasets import load_diabetes
import pandas as pd

# Load the Diabetes Dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display basic information about the dataset
print("Dataset Overview:")
print(df.head())

print("\nDataset Description:")
print(data.DESCR)
