import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
DB_PATH = '/Users/sachin/Downloads/regression.db'
TABLE_NAME = 'life_science'
 
    

def show_table_names(db_path=DB_PATH):
    """Prints every table inside the db (same as your first cell)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in cursor.fetchall():
        print(table[0])
    conn.close()
 
 
def load_data(db_path=DB_PATH, table_name=TABLE_NAME):
    """Connects to SQLite and returns the requested table as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'Select * from {table_name}', conn)
    conn.close()
    return df


def basic_info(df):
    """Your describe/info/shape/null-count checks, unchanged."""
    print(df.describe())
    print(df.info())
    print(df.columns)
    print(df.shape)
    print(df.isnull().sum())

def preprocess_data(df):
    """Cleans missing values in TCGA_DESC and PUTATIVE_TARGET."""
    df_clean = df.copy()
    if "TCGA_DESC" in df_clean.columns:
        df_clean["TCGA_DESC"] = df_clean["TCGA_DESC"].fillna(df_clean["TCGA_DESC"].mode()[0])
    if "PUTATIVE_TARGET" in df_clean.columns:
        df_clean["PUTATIVE_TARGET"] = df_clean["PUTATIVE_TARGET"].fillna(df_clean["PUTATIVE_TARGET"].mode()[0])
    return df_clean    

def plot_target_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["LN_IC50"], kde=True, bins=30, color="skyblue")
 
    mean = df["LN_IC50"].mean()
    median = df["LN_IC50"].median()
    minimum = df["LN_IC50"].min()
    maximum = df["LN_IC50"].max()
    skew = df["LN_IC50"].skew()
 
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    elif skew > 0:
        shape = "right-skewed"
    else:
        shape = "left-skewed"
 
    text = (
        f"Mean = {mean:.2f}\n"
        f"Median = {median:.2f}\n"
        f"Range = {minimum:.2f} to {maximum:.2f}\n"
        f"Distribution = {shape}"
    )
    plt.text(0.02, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
              verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.title("Distribution of LN_IC50")
    plt.xlabel("LN_IC50")
    plt.ylabel("Frequency")
    plt.show()
def plot_cancer_type_frequency(df):
    plt.figure(figsize=(10, 6))
    counts = df["TCGA_DESC"].value_counts()
    sns.countplot(y="TCGA_DESC", data=df, order=counts.index, palette="viridis")
 
    total = counts.sum()
    most_common = counts.index[0]
    most_common_count = counts.iloc[0]
    least_common = counts.index[-1]
    least_common_count = counts.iloc[-1]
 
    text = (
        f"Total samples = {total}\n"
        f"Cancer types = {len(counts)}\n"
        f"Most common = {most_common} ({most_common_count} samples)\n"
        f"Least common = {least_common} ({least_common_count} samples)"
    )
    plt.text(0.98, 0.02, text, transform=plt.gca().transAxes, fontsize=10,
              verticalalignment="bottom", horizontalalignment="right",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.title("Cancer Type Frequency")
    plt.xlabel("Number of Samples")
    plt.ylabel("Cancer Type")
    plt.show()



