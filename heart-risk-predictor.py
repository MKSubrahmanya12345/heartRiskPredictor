'''
code to check whether the required packages are installed.
else installs the packages
'''


import subprocess, sys

for package in ['pandas', 'numpy', 'sklearn', 'xgboost']:
    try:
        __import__(package)
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

print("step 1")

#importing the packages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score


df=pd.read_csv("heart.csv")


df = df.dropna()


# Replace cholesterol == 0 with mean cholesterol (excluding zeros)
df['cholesterol'] = df['cholesterol'].astype(float) 
mean_chol = df.loc[df['cholesterol'] != 0, 'cholesterol'].mean()
df.loc[df['cholesterol'] == 0, 'cholesterol'] = mean_chol





X = df.drop('target', axis=1)       # dropping target for training
y = df['target']                    # target is the required output and is separated from other columns

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# StandardScaler - to standardise the scales of values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(random_state=42)

# Hyperparameter tuning parameters grid
param_grid = {
    'n_estimators': [100],
    'max_depth': [5],
    'learning_rate': [0.1],
    'subsample': [0.8]
}

# GridSearchCV - Cross validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train_scaled, y_train)

# After finding the best parameters, the best model is used to predict
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Model output evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")





import numpy as np
import pandas as pd

# Step 1: Collect user input


# Step 1: User inputs (no prefilled values)
features = [
    'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    'fasting blood sugar', 'resting ecg', 'max heart rate',
    'exercise angina', 'oldpeak', 'ST slope'
]

user_data = {}
for feat in features:
    user_data[feat] = float(input(f"{feat}: "))

# Step 2: Save to CSV
input_df = pd.DataFrame([user_data])
input_df.to_csv("user_input.csv", index=False)

# Step 3: Reload and preprocess same as training
df_input = pd.read_csv("user_input.csv")

# Cholesterol fix (as done earlier)
df_input['cholesterol'] = df_input['cholesterol'].astype(float)
mean_chol = df['cholesterol'].loc[df['cholesterol'] != 0].mean()
df_input.loc[df_input['cholesterol'] == 0, 'cholesterol'] = mean_chol

# Apply scaler
input_scaled = scaler.transform(df_input)

# Step 4: Predict risk
risk_prob = best_model.predict_proba(input_scaled)[0][1]
print(f"\n⚠️  Estimated Heart Attack Risk: {round(risk_prob * 100, 2)}%")



# Step 1: Load first 10 rows from heart.csv
first_10 = df.head(10).drop('target', axis=1)  # remove 'target' since model predicts it

# Step 2: Save to new CSV
first_10.to_csv("sample_inputs.csv", index=False)

# Step 3: Reload and fix cholesterol
input_df = pd.read_csv("sample_inputs.csv")
input_df['cholesterol'] = input_df['cholesterol'].astype(float)
mean_chol = df['cholesterol'].loc[df['cholesterol'] != 0].mean()
input_df.loc[input_df['cholesterol'] == 0, 'cholesterol'] = mean_chol

# Step 4: Apply scaler
input_scaled = scaler.transform(input_df)

# Step 5: Predict
risk_probs = best_model.predict_proba(input_scaled)[:, 1]  # column 1 = probability of risk

# Step 6: Output results
for i, prob in enumerate(risk_probs):
    print(f"Sample {i+1} - ⚠️  Estimated Heart Attack Risk: {round(prob * 100, 2)}%")



import pandas as pd

# Create DataFrame from the given values
input_data = {
    'age': [49],
    'sex': [0],
    'chest pain type': [3],
    'resting bp s': [160],
    'cholesterol': [180],
    'fasting blood sugar': [0],
    'resting ecg': [0],
    'max heart rate': [156],
    'exercise angina': [0],
    'oldpeak': [1],
    'ST slope': [2]
}

input_df = pd.DataFrame(input_data)

# Save to CSV
input_df.to_csv("user_input.csv", index=False)

# Read back, clean cholesterol if needed
input_df = pd.read_csv("user_input.csv")
input_df['cholesterol'] = input_df['cholesterol'].astype(float)
mean_chol = df['cholesterol'].loc[df['cholesterol'] != 0].mean()
input_df.loc[input_df['cholesterol'] == 0, 'cholesterol'] = mean_chol

# Apply the same scaler
input_scaled = scaler.transform(input_df)

# Predict
risk_prob = best_model.predict_proba(input_scaled)[0][1]
risk_percentage = round(risk_prob * 100, 2)

print(f"\n⚠️  Estimated Heart Attack Risk: {risk_percentage}%")
