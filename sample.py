import pandas as pd

df = pd.read_csv("Patients.csv")


# #Day 1
# # First 5 rows
# print("========== HEAD ==========")
# print(df.head())

# # Last 5 rows
# print("\n========== TAIL ==========")
# print(df.tail())

# # Shape (Rows, Columns)
# print("\n========== SHAPE ==========")
# print(df.shape)

# # Column Names
# print("\n========== COLUMNS ==========")
# print(df.columns)

# # Dataset Information
# print("\n========== INFO ==========")
# df.info()

# # Statistical Summary
# print("\n========== DESCRIBE ==========")
# print(df.describe())

# # Single Column
# print("\n========== AGE COLUMN ==========")
# print(df["Age"])

# # Multiple Columns
# print("\n========== AGE & NAME ==========")
# print(df[["Age", "Name"]])

# # First Row
# print("\n========== FIRST ROW ==========")
# print(df.loc[0])

# # Patients Older than 50
# print("\n========== AGE > 50 ==========")
# print(df[df["Age"] > 50])

# # Average Age
# print("\n========== AVERAGE AGE ==========")
# print(df["Age"].mean())

# # Count Heart Disease
# print("\n========== HEART DISEASE COUNT ==========")
# print(df["HeartDisease"].value_counts())

# # Missing Values
# print("\n========== MISSING VALUES ==========")
# print(df.isnull().sum())

# # Sort by Age
# print("\n========== SORT BY AGE ==========")
# print(df.sort_values("Age"))


# #Day 2

# #Dropna (TO remove rows with missing values)

# print("\n========== DROPNA ==========")
# print(df.dropna())

# # Fillna (To fill missing values with a specific value)
# print("\n========== FILLNA ==========")
# df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].mean())
# print(df)

# # Find and drop duplicates
# print("\n========== DROP DUPLICATES ==========")
# print(df.duplicated())
# print(df.duplicated().sum())
# print(df.drop_duplicates())

# # Invalid Data
# print("\n========== INVALID DATA ==========")
# invalid_age = df[df["Age"] < 0]

# df.loc[df["Age"] == -10, "Age"] = 30

# print(df)

print(df.dtypes)

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

print(df)
print(df.isnull().sum())
