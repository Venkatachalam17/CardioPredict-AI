import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# Configure visualization style
sns.set_theme(style="whitegrid")

print("==================================================")
print("       CARDIOPREDICT: END-TO-END PIPELINE         ")
print("==================================================")

# --- STEP 1: DATA LOADING ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
response = requests.get(url, verify=False)

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal', 'target'
]

df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names, na_values='?')
print(f"\n[Step 1] Dataset Loaded Successfully! Shape: {df.shape}")

# --- STEP 2: DATA CLEANING & MISSING VALUES ---
print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print(f"\nDuplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

for col in ['ca', 'thal']:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' using median: {median_val}")

# Convert target into binary classification (0 = Healthy, 1 = At Risk)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
print("\nUpdated Target Class Distribution:")
print(df['target'].value_counts())

# --- STEP 3: OUTLIER DETECTION & TREATMENT (IQR Capping) ---
print("\n--- Running Outlier Treatment ---")
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

# Save Outlier Box Plots
plt.figure(figsize=(12, 6))
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.savefig('outlier_boxplots.png')
plt.close()
print("Saved visualization: 'outlier_boxplots.png'")

# --- STEP 4: CATEGORICAL ENCODING & FEATURE SCALING ---
print("\n--- Running Encoding & Feature Scaling ---")
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
print(f"Final Processed Feature Matrix Shape: {X.shape}")

# --- WEEK 3: MODEL BUILDING, TRAINING & EVALUATION ---
print("\n--- Starting Week 3: Model Training & Evaluation ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Baseline Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
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
print("Saved visualization: 'confusion_matrix.png'")

# --- HYPERPARAMETER TUNING (GridSearchCV) ---
print("\n--- Running Hyperparameter Tuning (GridSearchCV) ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best Hyperparameters Found:\n{grid_search.best_params_}")

best_rf_model = grid_search.best_estimator_
tuned_y_pred = best_rf_model.predict(X_test)
print(f"Tuned Model Accuracy: {accuracy_score(y_test, tuned_y_pred):.4f}")

# --- WEEK 4: EXPLAINABLE AI (XAI) & MODEL INTERPRETABILITY ---
print("\n--- Starting Week 4: Explainable AI (XAI) Analysis ---")

# 1. Gini Feature Importance
importances = best_rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Most Important Features (Gini Importance):")
print(feature_imp_df.head(5))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis')
plt.title('CardioPredict - Random Forest Feature Importance (Gini)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Saved visualization: 'feature_importance.png'")

# 2. Permutation Importance
perm_result = permutation_importance(best_rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Mean': perm_result.importances_mean
}).sort_values(by='Importance_Mean', ascending=False)

print("\nTop 5 Features by Permutation Importance:")
print(perm_df.head(5))

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance_Mean', y='Feature', data=perm_df.head(10), palette='magma')
plt.title('CardioPredict - Permutation Importance (Test Set)')
plt.xlabel('Mean Accuracy Drop')
plt.tight_layout()
plt.savefig('permutation_importance.png')
plt.close()
print("Saved visualization: 'permutation_importance.png'")

print("\n==================================================")
print(" PIPELINE EXECUTION COMPLETE SUCCESSFULLY!        ")
print("==================================================")