# Data Exploration Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = r"C:\Projects\hospital_readmission_prediction\output\cleaned_diabetic_data\ml_ready_data.csv"
df = pd.read_csv(data_path)

# Basic info
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.info())
print(df.describe())

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

# Class distribution
if 'readmitted' in df.columns:
    print("Readmitted value counts:")
    print(df['readmitted'].value_counts())
    sns.countplot(x='readmitted', data=df)
    plt.title("Readmission Distribution")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()