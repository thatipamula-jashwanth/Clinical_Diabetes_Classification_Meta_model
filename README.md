# Clinical Diabetes Classification using Ensemble Models and Meta-Model

## Project Overview
This project focuses on predicting diabetes using a clinical dataset with highly imbalanced classes (approx. 90% non-diabetic, 10% diabetic). We explored three base models — Random Forest, XGBoost, and LightGBM — and combined them using a Logistic Regression meta-model with stacking to improve overall performance.

---

## Dataset
- **Source:** Clinical dataset with patient features such as age, BMI, glucose levels, and categorical features like location.
- **Size:** 100,000 samples (80,000 train, 20,000 test)
- **Target Column:** `diabetes` (0 = Non-Diabetic, 1 = Diabetic)
- **Class Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| 0     | 91,500| 91.5%      |
| 1     | 8,500 | 8.5%       |

- **Preprocessing Steps:**
  - Removed unknown/NAN rows
  - One-hot encoded categorical features
  - Splitting into training and test sets
  - Used `class_weight='balanced'` to handle class imbalance

---

## Base Models

### 1. Random Forest
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Train   | 0.9175   | 0.5085    | 0.8940 | 0.6483   |
| Test    | 0.9183   | 0.5111    | 0.8812 | 0.6469   |

*Random Forest trained with:*  
- n_estimators = 100  
- max_depth = 10  
- min_samples_split = 5  
- min_samples_leaf = 3  
- class_weight = 'balanced'

**Visualizations:**  
- ROC Curve: `images/rf_ROC.png`  
- Confusion Matrix: `images/rf_confusion_matric.png`  
- Feature Importance: `images/rf_feature_imp.png`  

---

### 2. XGBoost
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Train   | 0.9267   | 0.5388    | 0.9578 | 0.6896   |
| Test    | 0.9188   | 0.5129    | 0.8912 | 0.6511   |

*XGBoost trained with hyperparameters:*  
- n_estimators = 100  
- max_depth = 10  
- learning_rate = 0.1  
- scale_pos_weight = 10  

**Visualizations:**  
- ROC Curve: `images/xg_ROC.png`  
- Confusion Matrix: `images/xg_confusion_matrix.png`  
- Feature Importance: `images/xg_feature_imp.png`

---

### 3. LightGBM
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Train   | 0.9240   | 0.5294    | 0.9522 | 0.6805   |
| Test    | 0.9170   | 0.5065    | 0.8947 | 0.6468   |

*LightGBM trained with hyperparameters:*  
- n_estimators = 100  
- max_depth = 10  
- learning_rate = 0.1  
- class_weight = 'balanced'  

**Visualizations:**  
- ROC Curve: `images/lgbm_roc.png`  
- Confusion Matrix: `images/lgbm_confusion_matrix.png`  
- Feature Importance: `images/lgbm_feature_imp.png`

---

## Meta-Model (Stacking)
- **Meta-learner:** Logistic Regression
- **Base models:** Random Forest, XGBoost, LightGBM
- **Cross-Validation:** 5-fold CV used for generating out-of-fold predictions to avoid data leakage

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Train   | 0.9143   | 0.498     | 0.9513 | 0.6538   |
| Test    | 0.9105   | 0.4861    | 0.9141 | 0.6347   |


