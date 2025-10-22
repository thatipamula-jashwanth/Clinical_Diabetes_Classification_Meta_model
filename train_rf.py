import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

rf = RandomForestClassifier(
    n_estimators=100,        
    max_depth=10,           
    min_samples_split=5,
    min_samples_leaf=3,      
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Random Forest training complete and saved as 'random_forest_model.pkl'!")
