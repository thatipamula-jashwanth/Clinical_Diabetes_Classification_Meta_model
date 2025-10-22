import pandas as pd

df = pd.read_csv("diabetes_dataset.csv")

print(df['diabetes'].value_counts())
print("\nPercentage distribution:")
print(df['diabetes'].value_counts(normalize=True) * 100)
