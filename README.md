Drug Sensitivity Prediction
This project uses machine learning to predict cancer drug sensitivity based on drug, cancer cell-line, and concentration-related features. The target variable, LN_IC50, represents the logarithmic drug concentration associated with cancer cell response.
What the Project Covers?
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature preparation
- Training and comparison of regression models
- Evaluation using RMSE, MAE, and R²
  Models Used
- Random Forest Regressor
- XGBoost Regressor
   Best Result — XGBoost
- RMSE: 1.2317
- MAE: 0.9212
- R²: 0.8032
The XGBoost model performed better and explained approximately 80% of the variation in LN_IC50 on the evaluated test data.
 Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Joblib
  Project Structure
├── data_loader.py
├── train.py
└── test_evaluate.py
Goal: Apply machine learning to cancer drug-response data and build a model capable of predicting drug sensitivity.
