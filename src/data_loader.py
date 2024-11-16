import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    # Load the dataset
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df
