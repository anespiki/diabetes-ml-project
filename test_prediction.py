#!/usr/bin/env python3
"""
Simple test script to demonstrate diabetes prediction with visualizations
Modify the patient_data below to test with your own patient information
"""

from predict_diabetes import DiabetesPredictor
from visualizer import DiabetesVisualizer

def test_patient_prediction():
    """Test diabetes prediction with custom patient data."""

    #LOW RISK PATIENT
    # patient_data = {
    #     'Pregnancies': 1,           # Number of pregnancies
    #     'Glucose': 120,            # Glucose level (mg/dL) - use 0 if unknown
    #     'BloodPressure': 80,       # Blood pressure (mmHg) - use 0 if unknown
    #     'SkinThickness': 25,       # Skin thickness (mm) - use 0 if unknown
    #     'Insulin': 0,              # Insulin level - use 0 if unknown (will be imputed)
    #     'BMI': 25.5,               # Body Mass Index - use 0 if unknown
    #     'DiabetesPedigreeFunction': 0.3,  # Diabetes pedigree function
    #     'Age': 30                  # Age in years
    # }

    #HIGH RISK PATIENT
    patient_data = {
          'Pregnancies': 6,
          'Glucose': 180,
          'BloodPressure': 90,
          'SkinThickness': 35,
          'Insulin': 0,
          'BMI': 35.0,
          'DiabetesPedigreeFunction': 0.8,
          'Age': 55
    }

    print("🏥 DIABETES RISK ASSESSMENT")
    print("=" * 50)

    try:
        # Initialize predictor
        print("Loading trained models...")
        predictor = DiabetesPredictor()

        # Make prediction
        result = predictor.predict_single_patient(patient_data)

        # Display results
        print(f"\n👤 PATIENT INFORMATION:")
        for key, value in patient_data.items():
            print(f"   {key}: {value}")

        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Overall Risk: {result['diabetes_risk']}")
        print(f"   Risk Probability: {result['risk_probability']}")

        # Show risk interpretation
        prob_float = float(result['risk_probability'])
        if prob_float >= 0.7:
            risk_level = "Very High Risk"
            emoji = "🔴"
        elif prob_float >= 0.5:
            risk_level = "High Risk"
            emoji = "🟡"
        elif prob_float >= 0.3:
            risk_level = "Moderate Risk"
            emoji = "🟠"
        else:
            risk_level = "Low Risk"
            emoji = "🟢"

        print(f"   Risk Level: {emoji} {risk_level}")

        # Show top 3 model predictions
        print(f"\n📊 TOP MODEL PREDICTIONS:")

        # Get model performance for ordering
        performance = predictor.get_model_performance()
        top_models = performance[:3] if performance else []

        for model_name, _ in top_models:
            if model_name in result['model_predictions']:
                pred_data = result['model_predictions'][model_name]
                print(f"   {model_name}: {pred_data['prediction']} ({pred_data['probability']})")

        print(f"\n💡 INTERPRETATION:")
        if prob_float >= 0.5:
            print("   ⚠️  This patient shows indicators suggesting diabetes risk.")
            print("   📋 Recommend further medical evaluation and testing.")
        else:
            print("   ✅ This patient shows lower diabetes risk indicators.")
            print("   🔄 Continue regular health monitoring and healthy lifestyle.")

        print(f"\n⚕️  MEDICAL DISCLAIMER:")
        print("   This is a research tool for educational purposes only.")
        print("   Always consult healthcare professionals for medical decisions.")

        # === GENERATE VISUALIZATIONS ===
        print(f"\n📊 GENERATING VISUALIZATIONS...")
        print("   Creating performance charts and prediction graphs...")

        try:
            # Initialize visualizer
            visualizer = DiabetesVisualizer()

            # Create comprehensive report with all visualizations
            visualizer.create_patient_summary_report(
                patient_data=patient_data,
                prediction_results=result,
                predictor=predictor,
                save=True,
                show=False
            )

            # Also create individual charts
            print("   📈 Creating individual model performance chart...")
            performance = predictor.get_model_performance()
            if performance:
                visualizer.plot_model_performance(performance, save=True, show=False)

            print("   🎯 Creating patient prediction charts...")
            visualizer.plot_patient_predictions(result, patient_data, save=True, show=False)

            print("   🔍 Creating feature importance charts...")
            visualizer.plot_feature_importance(predictor, save=True, show=False)

            print(f"\n✅ VISUALIZATIONS COMPLETE!")
            print("   📂 All charts saved in the 'graphs/' directory:")
            print("      • comprehensive_diabetes_report.png - Complete summary report")
            print("      • model_performance_comparison.png - Model ROC-AUC comparison")
            print("      • patient_predictions.png - Individual model predictions")
            print("      • feature_importance.png - Feature importance analysis")
            print("\n   💡 Open these files to view detailed visual analysis!")

        except Exception as viz_error:
            print(f"   ⚠️  Visualization Error: {viz_error}")
            print("   Predictions were successful, but graphs couldn't be generated.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you ran: ./diabetes_env/bin/python train_models.py")
        print("   2. Check that all patient data fields are provided")
        print("   3. Ensure virtual environment is set up correctly")

if __name__ == "__main__":
    test_patient_prediction()