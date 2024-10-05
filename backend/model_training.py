import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Load the dataset
df = pd.read_csv('dataset/loan_approval_dataset.csv')

# Preprocessing
df['loan_status'] = df['loan_status'].str.strip()
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

label_encoder = LabelEncoder()
df['education'] = label_encoder.fit_transform(df['education'])
df['self_employed'] = label_encoder.fit_transform(df['self_employed'])

# Features and target variable
X = df[['education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']]
y = df['loan_status']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# Store accuracies and predictions
accuracies = {}
y_preds = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[model_name] = acc
    y_preds[model_name] = y_pred
    print(f"{model_name} Accuracy: {acc:.2f}")

# Save the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
output_dir = 'backend/static'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/{best_model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# 1. Horizontal Bar Chart for Model Accuracies
plt.figure(figsize=(10, 6))
model_names = list(accuracies.keys())
model_accuracies = list(accuracies.values())
colors = ['#FF5733', '#33FF57', '#3357FF', '#F5B041']  # Different colors for each model

plt.barh(model_names, model_accuracies, color=colors)
plt.xlabel('Accuracy')
plt.title('Model Accuracies Comparison')
plt.xlim(0.9, 1.0)  # Adjust to focus on higher accuracies
plt.axvline(x=0.9, color='r', linestyle='--', label='90% Target')
plt.legend()
plt.grid(axis='x')

# Save the accuracy comparison plot
plt.savefig(f'{output_dir}/model_accuracies_comparison.png')
plt.close()

# 2. Confusion Matrices for Each Model
plt.figure(figsize=(12, 10))

for i, (model_name, y_pred) in enumerate(y_preds.items(), start=1):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.subplot(2, 2, i)
    
    # Confusion matrix display with integer format 'd'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')  # Use 'd' for integers
    plt.title(f'Confusion Matrix: {model_name}')

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrices.png')  # Save the confusion matrix as an image
plt.close()

# 3. Line Graph Comparison of Accuracies with distinct lines for each algorithm
plt.figure(figsize=(10, 6))

# Assign different colors and markers to each model
markers = ['o', 's', '^', 'D']
colors = ['#FF5733', '#33FF57', '#3357FF', '#F5B041']

# Plot each model's accuracy
for i, model_name in enumerate(model_names):
    plt.plot([0, 1], [model_accuracies[i], model_accuracies[i]], marker=markers[i], label=model_name, color=colors[i])

# Add details to the plot
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of ML Algorithms (Line Graph)')
plt.ylim(0.9, 1.0)  # Focus on higher accuracies
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Target')
plt.grid(True)
plt.legend()

# Save the line graph comparison plot
plt.savefig(f'{output_dir}/model_accuracies_line_graph.png')
plt.close()

# 4. ROC Curve for each model
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):  # Check if the model has predict_proba method
        y_proba = model.predict_proba(X_test)[:, 1]  # Get the probability for the positive class
    else:
        y_proba = model.decision_function(X_test)  # For models like SVM without predict_proba

    fpr, tpr, _ = roc_curve(y_test, y_proba)  # Calculate FPR and TPR
    roc_auc = auc(fpr, tpr)  # Calculate AUC
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')  # Plot the ROC curve for each model

# Add plot details
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Plot the diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend(loc='lower right')
plt.grid(True)

# Save the ROC curve plot
plt.savefig(f'{output_dir}/model_roc_curves.png')
plt.close()
