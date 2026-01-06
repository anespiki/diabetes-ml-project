#!/usr/bin/env python3
"""
Diabetes ML Model Training Script
Trains multiple machine learning models for diabetes prediction and saves them for later use.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# XGBoost and SMOTE
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(csv_path="diabetes.csv"):
    """Load and preprocess the diabetes dataset."""
    print("Loading and preprocessing data...")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Handle missing values (zeros that should be NaN)
    cols_with_zero_as_missing = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

    # Separate features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print(f"Missing values per column:\n{X.isna().sum()}")

    return X, y

def train_and_save_models(X, y, model_dir="models"):
    """Train multiple models and save them along with preprocessing objects."""

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Imputation
    print("Applying KNN imputation...")
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = knn_imputer.fit_transform(X_train)
    X_test_imputed = knn_imputer.transform(X_test)

    # Convert back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X.columns)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X.columns)

    # Scaling
    print("Applying standard scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Save preprocessing objects
    with open(os.path.join(model_dir, "knn_imputer.pkl"), "wb") as f:
        pickle.dump(knn_imputer, f)

    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Preprocessing objects saved.")

    # SMOTE for balanced training
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Model configurations
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression_smote": LogisticRegression(max_iter=1000, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "knn_smote": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "svm_smote": SVC(kernel="rbf", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42
        )
    }

    # Train and evaluate models
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Determine training data
        if "smote" in model_name:
            X_train_data = X_train_smote
            y_train_data = y_train_smote
        elif model_name in ["random_forest", "xgboost"]:
            # These use unscaled data
            X_train_data = X_train_imputed
            y_train_data = y_train
            X_test_data = X_test_imputed
        else:
            X_train_data = X_train_scaled
            y_train_data = y_train
            X_test_data = X_test_scaled

        # Train model
        model.fit(X_train_data, y_train_data)

        # Predict
        if model_name in ["random_forest", "xgboost"]:
            y_pred = model.predict(X_test_data)
            y_prob = model.predict_proba(X_test_data)[:, 1]
        else:
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # Save model
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        results[model_name] = {
            "model": model,
            "roc_auc": roc_auc,
            "predictions": y_pred,
            "probabilities": y_prob
        }

    # Save results summary
    results_summary = {name: {"roc_auc": res["roc_auc"]} for name, res in results.items()}
    with open(os.path.join(model_dir, "results_summary.pkl"), "wb") as f:
        pickle.dump(results_summary, f)

    # Save feature names for later use
    with open(os.path.join(model_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\n=== MODEL TRAINING COMPLETE ===")
    print(f"All models and preprocessing objects saved to: {model_dir}/")
    print("\nModel Performance Summary (ROC-AUC):")
    for name, score in sorted(results_summary.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        print(f"  {name}: {score['roc_auc']:.4f}")

    return results

def main():
    """Main training pipeline."""
    print("=== DIABETES ML MODEL TRAINING ===")
    print(f"Training started at: {datetime.now()}")

    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Train and save models
    results = train_and_save_models(X, y)

    print(f"\nTraining completed at: {datetime.now()}")
    print("Models are ready for prediction!")

if __name__ == "__main__":
    main()