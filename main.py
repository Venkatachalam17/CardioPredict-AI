import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure visualization style
sns.set_theme(style="whitegrid")

# 1. The official URL for the clean UCI Cleveland Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# 2. Fetch the data using requests (disabling strict SSL verification)
response = requests.get(url, verify=False)

# 3. Define the standard 14 column names
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal', 'target'
]

# 4. Load the text content into Pandas using io.StringIO
df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names, na_values='?')

print(f"Dataset Loaded Successfully! Shape: {df.shape}")

# --- STEP 2: DATA CLEANING & MISSING VALUES ---
print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print(f"\nDuplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Impute missing values using median
for col in ['ca', 'thal']:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' using median: {median_val}")

# Convert target into binary classification (0 = Healthy, 1 = At Risk)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print("\nUpdated Target Class Distribution:")
print(df['target'].value_counts())


# --- STEP 3: OUTLIER DETECTION & TREATMENT ---
print("\n--- Running Outlier Treatment ---")
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Optional: Uncomment the next 6 lines if you want the box plots to pop up visually
# plt.figure(figsize=(12, 6))
# for i, col in enumerate(numerical_cols, 1):
#     plt.subplot(2, 3, i)
#     sns.boxplot(y=df[col], color='skyblue')
#     plt.title(f'Box Plot of {col}')
# plt.tight_layout()
# plt.show()

# Function to cap outliers using IQR
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    print(f"Capped outliers for '{column}'")

for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    cap_outliers(df, col)

print("\nStep 3 Outlier Capping Complete!")


# --- STEP 4: CATEGORICAL ENCODING & FEATURE SCALING ---
from sklearn.preprocessing import StandardScaler

print("\n--- Running Encoding & Feature Scaling ---")

# 1. One-hot encode categorical variables with multiple categories
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 2. Separate features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# 3. Scale continuous numerical columns to normalize variance
scaler = StandardScaler()
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

print(f"Final Processed Feature Matrix Shape: {X.shape}")
print("\nPipeline execution complete! Ready for model training.")


# --- GENERATE & SAVE OUTLIER BOX PLOTS ---
print("\nGenerating and saving outlier box plots...")

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Box Plot of {col}')

plt.tight_layout()

# Save the plot as an image file in your project folder
plt.savefig('outlier_boxplots.png')
plt.close() # Close the plot to free up memory

print("Visualization saved successfully as 'outlier_boxplots.png' in your project folder!")
