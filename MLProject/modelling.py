import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
mlflow.set_experiment("Diabetes-Prediction-Basic")

print("Loading dataset...")
train_df = pd.read_csv('diabetes_preprocessing/diabetes_train.csv')
test_df = pd.read_csv('diabetes_preprocessing/diabetes_test.csv')

X_train = train_df.drop(columns=['diabetes'])
y_train = train_df['diabetes']
X_test = test_df.drop(columns=['diabetes'])
y_test = test_df['diabetes']

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print("\nMemulai training...")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nTraining selesai!")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Manual logging
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    print("Model tersimpan di MLflow!")