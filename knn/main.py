import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('IRIS.csv')

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['species'])
y = df['species']

# Split the dataset manually
train_size = 120
X_train, X_test = X.iloc[:train_size].to_numpy(), X.iloc[train_size:].to_numpy()
y_train, y_test = y.iloc[:train_size].to_numpy(), y.iloc[train_size:].to_numpy()


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
