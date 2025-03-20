import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('LifeExpectancy.csv')


train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

print(f"Training set size: {train_dataset.shape}")
print(f"Testing set size: {test_dataset.shape}")