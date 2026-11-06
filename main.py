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




# --- WEEK 3: MODEL BUILDING, TRAINING & EVALUATION ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("\n--- Starting Week 3: Model Training & Evaluation ---")

# 1. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 2. Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

# 3. Predict on the test set
y_pred = rf_model.predict(X_test)

# 4. Evaluate performance using standard classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Generate and save the Confusion Matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Healthy', 'At Risk'],
            yticklabels=['Healthy', 'At Risk'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('CardioPredict - Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion Matrix visualization saved successfully as 'confusion_matrix.png'!")