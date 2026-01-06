#!/usr/bin/env python3
"""
Standalone Visualization Generator
Creates graphs and charts for diabetes prediction system performance.
"""

from predict_diabetes import DiabetesPredictor
from visualizer import DiabetesVisualizer

def generate_all_visualizations():
    """Generate all available visualizations for the diabetes prediction system."""

    print("🎨 DIABETES PREDICTION SYSTEM - VISUALIZATION GENERATOR")
    print("=" * 60)

    try:
        # Initialize predictor and visualizer
        print("Loading trained models and initializing visualizer...")
        predictor = DiabetesPredictor()
        visualizer = DiabetesVisualizer()

        print("✅ Successfully loaded all components!")

        # Example patient for demonstration
        sample_patients = [
            {
                'name': 'Low Risk Patient',
                'data': {
                    'Pregnancies': 1,
                    'Glucose': 95,
                    'BloodPressure': 70,
                    'SkinThickness': 20,
                    'Insulin': 0,
                    'BMI': 22.0,
                    'DiabetesPedigreeFunction': 0.2,
                    'Age': 25
                }
            },
            {
                'name': 'High Risk Patient',
                'data': {
                    'Pregnancies': 6,
                    'Glucose': 180,
                    'BloodPressure': 90,
                    'SkinThickness': 35,
                    'Insulin': 0,
                    'BMI': 35.0,
                    'DiabetesPedigreeFunction': 0.8,
                    'Age': 55
                }
            }
        ]

        print(f"\n📊 GENERATING SYSTEM-WIDE VISUALIZATIONS...")

        # 1. Model Performance Comparison
        print("   📈 Creating model performance comparison...")
        performance = predictor.get_model_performance()
        if performance:
            visualizer.plot_model_performance(performance, save=True, show=False)

        # 2. Feature Importance Analysis
        print("   🎯 Creating feature importance analysis...")
        visualizer.plot_feature_importance(predictor, save=True, show=False)

        # 3. Generate reports for sample patients
        for i, patient in enumerate(sample_patients):
            print(f"   👤 Creating prediction visualizations for {patient['name']}...")

            # Make prediction
            result = predictor.predict_single_patient(patient['data'])

            # Create patient-specific visualizations
            visualizer.plot_patient_predictions(
                result, patient['data'],
                save=True, show=False
            )

            # Create comprehensive report
            # Modify output names to include patient type
            patient_output_dir = f"graphs/patient_{i+1}_{patient['name'].lower().replace(' ', '_')}"
            import os
            os.makedirs(patient_output_dir, exist_ok=True)
            visualizer.output_dir = patient_output_dir
            visualizer.create_patient_summary_report(
                patient_data=patient['data'],
                prediction_results=result,
                predictor=predictor,
                save=True,
                show=False
            )

            # Reset output directory
            visualizer.output_dir = "graphs"

            print(f"      Risk Level: {result['diabetes_risk']} ({result['risk_probability']})")

        print(f"\n✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print(f"📂 Charts saved in the following locations:")
        print(f"   • graphs/ - Main system visualizations")
        print(f"   • graphs/patient_1_low_risk_patient/ - Low risk patient example")
        print(f"   • graphs/patient_2_high_risk_patient/ - High risk patient example")
        print(f"\n📋 Available Charts:")
        print(f"   • model_performance_comparison.png - ROC-AUC scores for all models")
        print(f"   • feature_importance.png - Which patient features matter most")
        print(f"   • patient_predictions.png - Individual model predictions")
        print(f"   • comprehensive_diabetes_report.png - Complete analysis report")

        print(f"\n💡 USAGE TIPS:")
        print(f"   • Open the comprehensive reports for complete patient analysis")
        print(f"   • Compare low vs high risk patient visualizations")
        print(f"   • Use model performance chart to understand system reliability")
        print(f"   • Feature importance shows which measurements are most critical")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Ensure models are trained: ./diabetes_env/bin/python train_models.py")
        print(f"   2. Check matplotlib is installed in virtual environment")
        print(f"   3. Verify sufficient disk space for image files")

def main():
    """Main function."""
    generate_all_visualizations()

if __name__ == "__main__":
    main()