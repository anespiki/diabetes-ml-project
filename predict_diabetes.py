#!/usr/bin/env python3
"""
Diabetes Prediction Script
Uses trained machine learning models to predict diabetes risk for new patients.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Union, List

class DiabetesPredictor:
    """Class for making diabetes predictions using trained ML models."""

    def __init__(self, model_dir="models"):
        """Initialize the predictor by loading saved models and preprocessing objects."""
        self.model_dir = model_dir
        self.models = {}
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        self.results_summary = None

        self._load_preprocessing_objects()
        self._load_models()
        self._load_metadata()

    def _load_preprocessing_objects(self):
        """Load the imputer and scaler."""
        try:
            with open(os.path.join(self.model_dir, "knn_imputer.pkl"), "rb") as f:
                self.imputer = pickle.load(f)

            with open(os.path.join(self.model_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

            print("✓ Preprocessing objects loaded successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Preprocessing objects not found. Please run train_models.py first. {e}")

    def _load_models(self):
        """Load all trained models."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl') and
                      f not in ['knn_imputer.pkl', 'scaler.pkl', 'feature_names.pkl', 'results_summary.pkl']]

        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            try:
                with open(os.path.join(self.model_dir, model_file), "rb") as f:
                    self.models[model_name] = pickle.load(f)
                print(f"✓ Loaded model: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load model {model_name}: {e}")

        if not self.models:
            raise ValueError("No models found. Please run train_models.py first.")

    def _load_metadata(self):
        """Load feature names and results summary."""
        try:
            with open(os.path.join(self.model_dir, "feature_names.pkl"), "rb") as f:
                self.feature_names = pickle.load(f)

            with open(os.path.join(self.model_dir, "results_summary.pkl"), "rb") as f:
                self.results_summary = pickle.load(f)

            print("✓ Metadata loaded successfully.")
        except FileNotFoundError:
            print("⚠ Metadata files not found. Some functionality may be limited.")

    def _validate_input(self, patient_data):
        """Validate patient data format."""
        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        if isinstance(patient_data, dict):
            # Single patient
            missing_features = [f for f in required_features if f not in patient_data]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

        elif isinstance(patient_data, pd.DataFrame):
            # Multiple patients
            missing_features = [f for f in required_features if f not in patient_data.columns]
            if missing_features:
                raise ValueError(f"Missing required columns: {missing_features}")
        else:
            raise TypeError("Patient data must be a dictionary or pandas DataFrame")

    def _preprocess_data(self, patient_data):
        """Apply the same preprocessing as used in training."""
        # Convert to DataFrame if dictionary
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()

        # Ensure correct column order
        df = df[self.feature_names]

        # Handle missing values (zeros to NaN for specific columns)
        cols_with_zero_as_missing = [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
        ]
        for col in cols_with_zero_as_missing:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

        # Apply imputation
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns,
            index=df.index
        )

        # Apply scaling for models that need it
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_imputed),
            columns=df.columns,
            index=df.index
        )

        return df_imputed, df_scaled

    def predict(self, patient_data, return_probabilities=True, top_models=None):
        """
        Make diabetes predictions for patient(s).

        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Patient data with required features
        return_probabilities : bool
            Whether to return probability scores (default: True)
        top_models : int or None
            Number of best-performing models to use (default: all models)

        Returns:
        --------
        dict : Prediction results
        """
        # Validate input
        self._validate_input(patient_data)

        # Preprocess data
        df_imputed, df_scaled = self._preprocess_data(patient_data)

        # Select models to use
        if top_models and self.results_summary:
            sorted_models = sorted(
                self.results_summary.items(),
                key=lambda x: x[1]["roc_auc"],
                reverse=True
            )[:top_models]
            models_to_use = {name: self.models[name] for name, _ in sorted_models if name in self.models}
        else:
            models_to_use = self.models

        # Make predictions
        results = {
            "predictions": {},
            "probabilities": {} if return_probabilities else None,
            "consensus": {}
        }

        for model_name, model in models_to_use.items():
            try:
                # Choose appropriate data based on model type
                if model_name in ["random_forest", "xgboost"]:
                    X = df_imputed
                else:
                    X = df_scaled

                # Make predictions
                pred = model.predict(X)
                results["predictions"][model_name] = pred

                if return_probabilities:
                    prob = model.predict_proba(X)[:, 1]  # Probability of diabetes
                    results["probabilities"][model_name] = prob

            except Exception as e:
                print(f"⚠ Warning: Failed to get prediction from {model_name}: {e}")

        # Calculate consensus predictions
        if results["predictions"]:
            all_preds = np.array(list(results["predictions"].values()))
            consensus_pred = (np.mean(all_preds, axis=0) > 0.5).astype(int)
            results["consensus"]["prediction"] = consensus_pred

            if return_probabilities and results["probabilities"]:
                all_probs = np.array(list(results["probabilities"].values()))
                consensus_prob = np.mean(all_probs, axis=0)
                results["consensus"]["probability"] = consensus_prob

        return results

    def predict_single_patient(self, patient_data):
        """
        Convenient method for single patient prediction with formatted output.

        Parameters:
        -----------
        patient_data : dict
            Single patient data

        Returns:
        --------
        dict : Formatted prediction results
        """
        results = self.predict(patient_data)

        # Format results for single patient
        formatted_results = {
            "patient_data": patient_data,
            "diabetes_risk": "HIGH" if results["consensus"]["prediction"][0] == 1 else "LOW",
            "risk_probability": f"{results['consensus']['probability'][0]:.3f}" if results["consensus"].get("probability") is not None else "N/A",
            "model_predictions": {}
        }

        # Add individual model results
        for model_name in results["predictions"]:
            pred = results["predictions"][model_name][0]
            prob = results["probabilities"][model_name][0] if results["probabilities"] else None

            formatted_results["model_predictions"][model_name] = {
                "prediction": "Diabetes" if pred == 1 else "No Diabetes",
                "probability": f"{prob:.3f}" if prob is not None else "N/A"
            }

        return formatted_results

    def get_model_performance(self):
        """Get performance summary of all models."""
        if self.results_summary:
            return sorted(
                self.results_summary.items(),
                key=lambda x: x[1]["roc_auc"],
                reverse=True
            )
        else:
            return None

def main():
    """Example usage of the DiabetesPredictor."""
    try:
        # Initialize predictor
        predictor = DiabetesPredictor()

        # Example patient data
        sample_patient = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,  # Will be imputed
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }

        print("\n=== DIABETES PREDICTION EXAMPLE ===")
        print(f"Patient Data: {sample_patient}")

        # Make prediction
        result = predictor.predict_single_patient(sample_patient)

        print(f"\n🔍 PREDICTION RESULTS:")
        print(f"Overall Risk Assessment: {result['diabetes_risk']}")
        print(f"Risk Probability: {result['risk_probability']}")

        print(f"\n📊 Individual Model Predictions:")
        for model_name, pred_data in result['model_predictions'].items():
            print(f"  {model_name}: {pred_data['prediction']} (prob: {pred_data['probability']})")

        # Show model performance
        performance = predictor.get_model_performance()
        if performance:
            print(f"\n⭐ Model Performance (ROC-AUC):")
            for model_name, metrics in performance:
                print(f"  {model_name}: {metrics['roc_auc']:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease run 'python train_models.py' first to train and save models.")

if __name__ == "__main__":
    main()