import pandas as pd

df = pd.read_csv("diabetes_dataset.csv")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", categorical_cols)

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded.to_csv("clinical_diabetes_encoded.csv", index=False)
print("Saved encoded dataset to 'clinical_diabetes_encoded.csv'")
