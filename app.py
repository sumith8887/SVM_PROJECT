import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.datasets import load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

st.set_page_config("End-to-End SVM", layout="wide")
st.title("End-to-End SVM Platform (Classification & Regression)")

st.sidebar.header("SVM Settings")
problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

st.header("Step 1 : Data Ingestion")

@st.cache_data
def load_data():
    data = load_diabetes(as_frame=True)
    df = data.frame
    df['target'] = data.target
    np.random.seed(42)
    for col in df.columns[:-1]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan
    return df

df = load_data()
raw_path = os.path.join(RAW_DIR, "diabetes_raw.csv")
df.to_csv(raw_path, index=False)

st.success("Diabetes Dataset Loaded")
st.info(f"Raw dataset saved at: {raw_path}")
st.dataframe(df.head())

st.header("Step 2 : Exploratory Data Analysis")
st.write("Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())

fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.header("Step 3 : Data Cleaning")

strategy = st.selectbox(
    "Missing Value Strategy",
    ["Mean", "Median", "Drop Rows"]
)

# Create a copy of original dataframe
df_clean = df.copy()

# Handle missing values
if strategy == "Drop Rows":
    df_clean = df_clean.dropna()

elif strategy == "Mean":
    # Fill NaN with column mean
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].mean()
    )

elif strategy == "Median":
    # Fill NaN with column median
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].median()
    )

# Display cleaned dataset info
st.subheader("Cleaned Dataset Preview")
st.dataframe(df_clean.head())

st.subheader("Remaining Missing Values")
st.write(df_clean.isnull().sum())

# Store in session state
st.session_state.df_clean = df_clean

st.success("Data Cleaning Completed Successfully")

# Save cleaned dataset
if st.button("Save Cleaned Dataset"):
    clean_path = os.path.join(CLEAN_DIR, "cleaned_diabetes.csv")
    df_clean.to_csv(clean_path, index=False)

    st.success("Cleaned Dataset Saved Successfully")
    st.info(f"Saved at: {clean_path}")

st.header("Step 4 : Load Cleaned Dataset")
clean_files = [f for f in os.listdir(CLEAN_DIR) if "diabetes" in f.lower()]

if not clean_files:
    st.error("No cleaned dataset found. Please save one in Step 4.")
    st.stop()

selected = st.selectbox("Select Cleaned Dataset", clean_files)
df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))
st.dataframe(df_model.head())

st.header("Step 5 : Train SVM")
target = "target"
X = df_model.drop(columns=[target])
y = df_model[target]

if problem_type == "Classification":
    threshold = y.median()
    st.info(f"Binary Classification → High Value ≥ {threshold:.2f}")
    y = (y >= threshold).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

if problem_type == "Classification":
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
else:
    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.success(f"R² Score: {r2:.2f}")
    st.success(f"MSE: {mse:.2f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
