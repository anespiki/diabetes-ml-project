#!/usr/bin/env python3
"""
Diabetes Prediction Visualizer
Creates graphs and charts to visualize model performance and prediction results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge
import os

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class DiabetesVisualizer:
    """Class for creating visualizations of diabetes prediction results."""

    def __init__(self, output_dir="graphs"):
        """Initialize visualizer with output directory for saving graphs."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_performance(self, performance_data, save=True, show=False):
        """
        Create bar chart comparing model performance (ROC-AUC scores).

        Parameters:
        -----------
        performance_data : list of tuples
            [(model_name, {"roc_auc": score}), ...]
        save : bool
            Whether to save the plot
        show : bool
            Whether to display the plot
        """
        if not performance_data:
            print("No performance data available for plotting.")
            return

        # Extract model names and scores
        models = [item[0] for item in performance_data]
        scores = [item[1]["roc_auc"] for item in performance_data]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create color map based on performance
        colors = plt.cm.RdYlGn([score for score in scores])

        # Create bar plot
        bars = plt.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Customize plot
        plt.title('Model Performance Comparison\n(ROC-AUC Scores)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
        plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.0)

        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add horizontal line at 0.8 (good performance threshold)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good Performance (0.8)')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance (0.5)')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        if save:
            filepath = os.path.join(self.output_dir, "model_performance_comparison.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📊 Model performance chart saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_patient_predictions(self, prediction_results, patient_data, save=True, show=False):
        """
        Create visualization showing individual model predictions for a patient.

        Parameters:
        -----------
        prediction_results : dict
            Results from DiabetesPredictor.predict_single_patient()
        patient_data : dict
            Original patient data
        save : bool
            Whether to save the plot
        show : bool
            Whether to display the plot
        """
        model_predictions = prediction_results.get('model_predictions', {})
        if not model_predictions:
            print("No model predictions available for plotting.")
            return

        # Extract data for plotting
        models = list(model_predictions.keys())
        probabilities = [float(pred['probability']) for pred in model_predictions.values()]
        predictions = [pred['prediction'] for pred in model_predictions.values()]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Probability bars
        colors = ['red' if prob >= 0.5 else 'green' for prob in probabilities]
        bars = ax1.bar(models, probabilities, color=colors, alpha=0.7, edgecolor='black')

        ax1.set_title('Individual Model Probability Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Diabetes Probability', fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')

        # Add probability labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Risk Gauge for consensus prediction
        consensus_prob = float(prediction_results.get('risk_probability', 0))
        self._create_risk_gauge(ax2, consensus_prob)

        plt.tight_layout()

        # Save plot
        if save:
            filepath = os.path.join(self.output_dir, "patient_predictions.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📈 Patient predictions chart saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def _create_risk_gauge(self, ax, probability):
        """Create a risk gauge/meter visualization."""
        # Create semicircle gauge
        theta1, theta2 = 0, 180

        # Define risk zones
        zones = [
            (0, 0.3, 'green', 'Low Risk'),
            (0.3, 0.5, 'yellow', 'Moderate Risk'),
            (0.5, 0.7, 'orange', 'High Risk'),
            (0.7, 1.0, 'red', 'Very High Risk')
        ]

        # Draw gauge zones
        for start, end, color, label in zones:
            start_angle = 180 - (start * 180)
            end_angle = 180 - (end * 180)
            wedge = Wedge((0, 0), 1, end_angle, start_angle,
                         facecolor=color, alpha=0.3, edgecolor='black')
            ax.add_patch(wedge)

        # Draw needle
        angle = 180 - (probability * 180)
        needle_x = 0.8 * np.cos(np.radians(angle))
        needle_y = 0.8 * np.sin(np.radians(angle))
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.05,
                fc='black', ec='black', linewidth=3)

        # Add center circle
        circle = plt.Circle((0, 0), 0.1, facecolor='black')
        ax.add_patch(circle)

        # Customize gauge
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Add title and probability text
        ax.text(0, -0.15, f'Risk Probability: {probability:.1%}',
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, 1.1, 'Diabetes Risk Gauge',
               ha='center', va='center', fontsize=14, fontweight='bold')

        # Add risk zone labels
        ax.text(-0.9, 0.7, 'Low\nRisk', ha='center', va='center', fontweight='bold', color='darkgreen')
        ax.text(-0.3, 0.9, 'Moderate', ha='center', va='center', fontweight='bold', color='darkorange')
        ax.text(0.3, 0.9, 'High', ha='center', va='center', fontweight='bold', color='darkorange')
        ax.text(0.9, 0.7, 'Very High\nRisk', ha='center', va='center', fontweight='bold', color='darkred')

    def plot_feature_importance(self, predictor, save=True, show=False):
        """
        Plot feature importance from the best performing models.

        Parameters:
        -----------
        predictor : DiabetesPredictor
            Trained predictor instance
        save : bool
            Whether to save the plot
        show : bool
            Whether to display the plot
        """
        try:
            # Get Random Forest and XGBoost models (they have feature importance)
            rf_model = predictor.models.get('random_forest')
            xgb_model = predictor.models.get('xgboost')

            if not rf_model and not xgb_model:
                print("No models with feature importance available.")
                return

            feature_names = predictor.feature_names
            if not feature_names:
                print("Feature names not available.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            plot_count = 0

            # Plot Random Forest importance
            if rf_model:
                importances_rf = rf_model.feature_importances_
                indices = np.argsort(importances_rf)[::-1]

                axes[plot_count].bar(range(len(importances_rf)),
                                   importances_rf[indices], color='forestgreen', alpha=0.7)
                axes[plot_count].set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
                axes[plot_count].set_xlabel('Features', fontsize=12)
                axes[plot_count].set_ylabel('Importance Score', fontsize=12)
                axes[plot_count].set_xticks(range(len(importances_rf)))
                axes[plot_count].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                axes[plot_count].grid(axis='y', alpha=0.3)
                plot_count += 1

            # Plot XGBoost importance
            if xgb_model:
                importances_xgb = xgb_model.feature_importances_
                indices = np.argsort(importances_xgb)[::-1]

                axes[plot_count].bar(range(len(importances_xgb)),
                                   importances_xgb[indices], color='orange', alpha=0.7)
                axes[plot_count].set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
                axes[plot_count].set_xlabel('Features', fontsize=12)
                axes[plot_count].set_ylabel('Importance Score', fontsize=12)
                axes[plot_count].set_xticks(range(len(importances_xgb)))
                axes[plot_count].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                axes[plot_count].grid(axis='y', alpha=0.3)
                plot_count += 1

            # Hide unused subplot if only one model available
            if plot_count == 1:
                axes[1].axis('off')

            plt.tight_layout()

            # Save plot
            if save:
                filepath = os.path.join(self.output_dir, "feature_importance.png")
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"🎯 Feature importance chart saved: {filepath}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error creating feature importance plot: {e}")

    def create_patient_summary_report(self, patient_data, prediction_results, predictor, save=True, show=False):
        """
        Create a comprehensive summary report with multiple visualizations.

        Parameters:
        -----------
        patient_data : dict
            Original patient data
        prediction_results : dict
            Results from prediction
        predictor : DiabetesPredictor
            Trained predictor instance
        save : bool
            Whether to save the plot
        show : bool
            Whether to display the plot
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))

        # Define grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Patient Info Table
        ax1 = fig.add_subplot(gs[0, :])
        self._create_patient_info_table(ax1, patient_data, prediction_results)

        # 2. Model Performance Comparison
        ax2 = fig.add_subplot(gs[1, :2])
        performance = predictor.get_model_performance()
        if performance:
            models = [item[0] for item in performance]
            scores = [item[1]["roc_auc"] for item in performance]
            colors = plt.cm.RdYlGn([score for score in scores])
            bars = ax2.bar(models, scores, color=colors, alpha=0.8, edgecolor='black')
            ax2.set_title('Model Performance (ROC-AUC)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('ROC-AUC Score')
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

            # Add score labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. Risk Gauge
        ax3 = fig.add_subplot(gs[1, 2])
        consensus_prob = float(prediction_results.get('risk_probability', 0))
        self._create_risk_gauge(ax3, consensus_prob)

        # 4. Individual Model Predictions
        ax4 = fig.add_subplot(gs[2, :])
        model_predictions = prediction_results.get('model_predictions', {})
        if model_predictions:
            models = list(model_predictions.keys())
            probabilities = [float(pred['probability']) for pred in model_predictions.values()]
            colors = ['red' if prob >= 0.5 else 'green' for prob in probabilities]
            bars = ax4.bar(models, probabilities, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_title('Individual Model Predictions', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Diabetes Probability')
            ax4.set_ylim(0, 1.0)
            ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(axis='y', alpha=0.3)

            # Add probability labels
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

        # 5. Feature Importance (if available)
        ax5 = fig.add_subplot(gs[3, :2])
        xgb_model = predictor.models.get('xgboost')
        if xgb_model and predictor.feature_names:
            importances = xgb_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax5.bar(range(len(importances)), importances[indices], color='orange', alpha=0.7)
            ax5.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Importance Score')
            ax5.set_xticks(range(len(importances)))
            ax5.set_xticklabels([predictor.feature_names[i] for i in indices], rotation=45, ha='right')
            ax5.grid(axis='y', alpha=0.3)

        # 6. Risk Distribution Context
        ax6 = fig.add_subplot(gs[3, 2])
        self._create_risk_context_plot(ax6, consensus_prob)

        # Overall title
        fig.suptitle('Diabetes Risk Assessment Report', fontsize=20, fontweight='bold', y=0.98)

        # Save comprehensive report
        if save:
            filepath = os.path.join(self.output_dir, "comprehensive_diabetes_report.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📋 Comprehensive report saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def _create_patient_info_table(self, ax, patient_data, prediction_results):
        """Create a formatted table showing patient information and results."""
        ax.axis('off')

        # Create patient info text
        risk_level = prediction_results.get('diabetes_risk', 'Unknown')
        probability = prediction_results.get('risk_probability', 'N/A')

        # Format patient data for display
        info_text = f"""
        PATIENT INFORMATION & RESULTS

        Age: {patient_data.get('Age', 'N/A')} years          Pregnancies: {patient_data.get('Pregnancies', 'N/A')}          BMI: {patient_data.get('BMI', 'N/A')}
        Glucose: {patient_data.get('Glucose', 'N/A')} mg/dL      Blood Pressure: {patient_data.get('BloodPressure', 'N/A')} mmHg      Skin Thickness: {patient_data.get('SkinThickness', 'N/A')} mm
        Insulin: {patient_data.get('Insulin', 'N/A')}           Diabetes Pedigree: {patient_data.get('DiabetesPedigreeFunction', 'N/A')}

        PREDICTION RESULTS:
        Overall Risk Level: {risk_level}          Risk Probability: {probability}
        """

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

    def _create_risk_context_plot(self, ax, patient_probability):
        """Create a plot showing where patient falls in risk distribution."""
        # Simulate population risk distribution (this would ideally come from real data)
        population_risks = np.random.beta(2, 5, 1000)  # Skewed towards lower risk

        ax.hist(population_risks, bins=30, alpha=0.6, color='lightblue',
               edgecolor='black', density=True, label='Population Distribution')

        # Mark patient's risk
        ax.axvline(patient_probability, color='red', linestyle='--', linewidth=3,
                  label=f'Patient Risk: {patient_probability:.1%}')

        ax.set_title('Risk in Population Context', fontsize=12, fontweight='bold')
        ax.set_xlabel('Diabetes Risk Probability')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

def main():
    """Example usage of visualizer."""
    # This would typically be called from the prediction script
    print("DiabetesVisualizer - Use this module with predict_diabetes.py")

if __name__ == "__main__":
    main()