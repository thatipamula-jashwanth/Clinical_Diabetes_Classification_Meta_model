import pandas as pd
from lightgbm import LGBMClassifier
import pickle
import re

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

def clean_col(col_name):
    return re.sub(r'[^0-9a-zA-Z_]', '_', col_name)

X_train.columns = [clean_col(c) for c in X_train.columns]
X_test.columns = [clean_col(c) for c in X_test.columns]

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)

lgbm.fit(X_train, y_train)

with open("lightgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm, f)

print("LightGBM training complete and saved as 'lightgbm_model.pkl'!")
