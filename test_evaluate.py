#!/usr/bin/env python
# coding: utf-8

# In[76]:


# train.py

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
)

from data_loader import load_data, preprocess_data

def encode_and_scale_features(X):
    """Encodes categorical columns and scales numerical features."""
    encoders = {}
    X_encoded = X.copy()

    cat_cols = X_encoded.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    joblib.dump(encoders, 'encoders.pkl')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    joblib.dump(scaler, 'scaler.pkl')

    return X_encoded, X_scaled

def train_logistic_regression(X_train_scaled, y_train_binary):
    """Trains a Logistic Regression model for classification."""
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train_binary)
    return log_reg

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Regressor model."""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    """Trains an XGBoost Regressor model."""
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def print_classification_diagnostics(model, X_val, y_val, model_name="Logistic Regression"):
    """Prints comprehensive classification evaluation metrics."""
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n================ {model_name} Evaluation ================")
    print(f"Accuracy:         {accuracy_score(y_val, preds):.4f}")
    print(f"Precision:        {precision_score(y_val, preds):.4f}")
    print(f"Recall:           {recall_score(y_val, preds):.4f}")
    print(f"F1-Score:         {f1_score(y_val, preds):.4f}")
    if probs is not None:
        print(f"ROC-AUC Score:    {roc_auc_score(y_val, probs):.4f}")

    cm = confusion_matrix(y_val, preds)
    print("\nConfusion Matrix:")
    print(f"  [ TN: {cm[0][0]:<6} FP: {cm[0][1]:<6} ]")
    print(f"  [ FN: {cm[1][0]:<6} TP: {cm[1][1]:<6} ]")

def print_regression_diagnostics(model, X_val, y_val, feature_names, model_name="Regressor"):
    """Prints comprehensive regression evaluation metrics and top feature importances."""
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    exp_var = explained_variance_score(y_val, preds)

    print(f"\n================ {model_name} Evaluation ================")
    print(f"R² Score:           {r2:.4f}")
    print(f"Explained Variance: {exp_var:.4f}")
    print(f"RMSE:               {rmse:.4f}")
    print(f"MAE:                {mae:.4f}")



def run_training_pipeline():
    """Executes full training pipeline with detailed evaluation diagnostics."""
    raw_df = load_data()
    df_clean = preprocess_data(raw_df)

    y_reg = df_clean['LN_IC50']
    y_bin = (y_reg > y_reg.median()).astype(int)

    drop_cols = ['LN_IC50', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID', 'SANGER_MODEL_ID']
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    feature_names = X.columns.tolist()

    X_encoded, X_scaled = encode_and_scale_features(X)

    X_tr, X_val, y_tr_reg, y_val_reg = train_test_split(X_encoded, y_reg, test_size=0.2, random_state=42)
    X_tr_sc, X_val_sc, y_tr_bin, y_val_bin = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    log_reg = train_logistic_regression(X_tr_sc, y_tr_bin)
    joblib.dump(log_reg, 'logistic_regression.pkl')
    print_classification_diagnostics(log_reg, X_val_sc, y_val_bin, "Logistic Regression")

    # 2. Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf_model = train_random_forest(X_tr, y_tr_reg)
    joblib.dump(rf_model, 'rf_model.pkl')
    print_regression_diagnostics(rf_model, X_val, y_val_reg, feature_names, "Random Forest")

    # 3. XGBoost Regressor
    print("\nTraining XGBoost Regressor...")
    xgb_model = train_xgboost(X_tr, y_tr_reg)
    joblib.dump(xgb_model, 'xgb_model.pkl')
    print_regression_diagnostics(xgb_model, X_val, y_val_reg, feature_names, "XGBoost")

    print("\n==================================================")
    print("All models and metrics successfully executed & saved!")
    print("==================================================")
    return log_reg, rf_model, xgb_model, X_tr

if __name__ == "__main__":

    run_training_pipeline()


# In[80]:


import matplotlib.pyplot as plt
import xgboost as xgb
#  loading trained model from disk
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# extract features names from datasets
raw_df = load_data()
df_clean = preprocess_data(raw_df)
drop_cols = ['LN_IC50', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID', 'SANGER_MODEL_ID']
feature_names = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns]).columns

# 3. Plot Random Forest Feature Importance
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
plt.barh(feature_names, importances)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 4. Plot XGBoost Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, importance_type="gain")
plt.title("XGBoost Model Best Features")
plt.tight_layout()
plt.show()


# In[82]:


get_ipython().system('jupyter nbconvert --to script Training_data.ipynb --output train')


# In[ ]:





# In[ ]:




