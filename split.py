import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("clinical_diabetes_encoded.csv")


X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Saved X_train, X_test, y_train, y_test as separate CSV files.")
