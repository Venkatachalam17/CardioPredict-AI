import pandas as pd

df = pd.read_csv("Patients.csv")

# First 5 rows
print("========== HEAD ==========")
print(df.head())

# Last 5 rows
print("\n========== TAIL ==========")
print(df.tail())

# Shape (Rows, Columns)
print("\n========== SHAPE ==========")
print(df.shape)

# Column Names
print("\n========== COLUMNS ==========")
print(df.columns)

# Dataset Information
print("\n========== INFO ==========")
df.info()

# Statistical Summary
print("\n========== DESCRIBE ==========")
print(df.describe())

# Single Column
print("\n========== AGE COLUMN ==========")
print(df["Age"])

# Multiple Columns
print("\n========== AGE & NAME ==========")
print(df[["Age", "Name"]])

# First Row
print("\n========== FIRST ROW ==========")
print(df.loc[0])

# Patients Older than 50
print("\n========== AGE > 50 ==========")
print(df[df["Age"] > 50])

# Average Age
print("\n========== AVERAGE AGE ==========")
print(df["Age"].mean())

# Count Heart Disease
print("\n========== HEART DISEASE COUNT ==========")
print(df["HeartDisease"].value_counts())

# Missing Values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Sort by Age
print("\n========== SORT BY AGE ==========")
print(df.sort_values("Age"))




