import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


with open("random_forest_model.pkl", "rb") as f:
    rf = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    xgb = pickle.load(f)

with open("lightgbm_model.pkl", "rb") as f:
    lgbm = pickle.load(f)

with open("stacked_meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)


X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


def align_columns(model_columns, X):
    """Ensure X has the same columns as model, filling missing with 0"""
    return X.reindex(columns=model_columns, fill_value=0)

def generate_meta_features(rf, xgb, lgbm, X):
    # RF
    if hasattr(rf, "feature_names_in_"):
        X_rf = align_columns(rf.feature_names_in_, X)
    else:
        X_rf = X
    rf_pred = rf.predict_proba(X_rf)[:,1]

    
    if hasattr(xgb, "get_booster"):
        X_xgb = align_columns(xgb.get_booster().feature_names, X)
    else:
        X_xgb = X
    xgb_pred = xgb.predict_proba(X_xgb)[:,1]

    
    if hasattr(lgbm, "booster_"):
        X_lgbm = align_columns(lgbm.booster_.feature_name(), X)
    else:
        X_lgbm = X
    lgbm_pred = lgbm.predict_proba(X_lgbm)[:,1]

    return pd.DataFrame({
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "lgbm_pred": lgbm_pred
    })


X_meta_train = generate_meta_features(rf, xgb, lgbm, X_train)
X_meta_test = generate_meta_features(rf, xgb, lgbm, X_test)


y_train_pred = meta_model.predict(X_meta_train)
y_test_pred = meta_model.predict(X_meta_test)


def print_metrics(y_true, y_pred, name="Set"):
    print(f"--- {name} Performance ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1-score :", round(f1_score(y_true, y_pred), 4))
    print()

print_metrics(y_train, y_train_pred, "Train")
print_metrics(y_test, y_test_pred, "Test")
