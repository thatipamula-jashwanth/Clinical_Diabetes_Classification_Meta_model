import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


with open("random_forest_model.pkl", "rb") as f:
    rf = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    xgb = pickle.load(f)

with open("lightgbm_model.pkl", "rb") as f:
    lgbm = pickle.load(f)


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()


def align_columns(model_columns, X):
    """Ensure X has the same columns as model, filling missing with 0"""
    return X.reindex(columns=model_columns, fill_value=0)

def get_oof_preds(model, X, y, n_splits=5):
    oof_preds = np.zeros(X.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

   
        if hasattr(model, "feature_names_in_"):
            X_tr_aligned = align_columns(model.feature_names_in_, X_tr)
            X_val_aligned = align_columns(model.feature_names_in_, X_val)
        elif hasattr(model, "get_booster"): 
            X_tr_aligned = align_columns(model.get_booster().feature_names, X_tr)
            X_val_aligned = align_columns(model.get_booster().feature_names, X_val)
        elif hasattr(model, "booster_"):  
            X_tr_aligned = align_columns(model.booster_.feature_name(), X_tr)
            X_val_aligned = align_columns(model.booster_.feature_name(), X_val)
        else:
            X_tr_aligned, X_val_aligned = X_tr, X_val

        model.fit(X_tr_aligned, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val_aligned)[:, 1]

    return oof_preds


print("Generating OOF predictions for RF...")
rf_oof = get_oof_preds(rf, X_train, y_train)

print("Generating OOF predictions for XGB...")
xgb_oof = get_oof_preds(xgb, X_train, y_train)

print("Generating OOF predictions for LGBM...")
lgbm_oof = get_oof_preds(lgbm, X_train, y_train)

meta_features = pd.DataFrame({
    "rf_pred": rf_oof,
    "xgb_pred": xgb_oof,
    "lgbm_pred": lgbm_oof
})


meta_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
meta_model.fit(meta_features, y_train)


with open("stacked_meta_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)

print("Stacked meta-model trained with CV and saved as 'stacked_meta_model.pkl'!")
