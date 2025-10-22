import pandas as pd
from xgboost import XGBClassifier
import pickle

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


xgb = XGBClassifier(
    n_estimators=200,    
    max_depth=6,            
    learning_rate=0.1,      
    subsample=0.8,          
    colsample_bytree=0.8,   
    min_child_weight=5,   
    scale_pos_weight=scale_pos_weight, 
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)


xgb.fit(X_train, y_train)


with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("XGBoost training complete and saved as 'xgboost_model.pkl'!")
