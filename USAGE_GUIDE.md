# Diabetes Prediction System - Usage Guide

This guide shows you how to use the diabetes prediction system to predict diabetes risk for new patients.

## Quick Start

### 1. Set Up Environment
```bash
# Create virtual environment (first time only)
python3 -m venv diabetes_env

# Activate virtual environment
source diabetes_env/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)
```bash
# Train and save all models
./diabetes_env/bin/python train_models.py
```

### 3. Make Predictions
```bash
# Test with example patient
./diabetes_env/bin/python predict_diabetes.py

# Test with visualizations (recommended)
./diabetes_env/bin/python test_prediction.py
```

## Making Predictions for New Patients

### Method 1: Modify the Example Script

Edit `predict_diabetes.py` and change the `sample_patient` dictionary:

```python
# Your patient's data
sample_patient = {
    'Pregnancies': 2,           # Number of pregnancies
    'Glucose': 120,            # Glucose level (mg/dL)
    'BloodPressure': 80,       # Blood pressure (mmHg)
    'SkinThickness': 25,       # Skin thickness (mm)
    'Insulin': 0,              # Insulin level (0 = missing, will be imputed)
    'BMI': 28.5,               # Body Mass Index
    'DiabetesPedigreeFunction': 0.5,  # Diabetes pedigree function
    'Age': 35                  # Age in years
}
```

### Method 2: Use as Python Module

Create your own script:

```python
from predict_diabetes import DiabetesPredictor

# Initialize predictor
predictor = DiabetesPredictor()

# Patient data
patient_data = {
    'Pregnancies': 1,
    'Glucose': 130,
    'BloodPressure': 90,
    'SkinThickness': 30,
    'Insulin': 0,  # Missing values (0) will be automatically imputed
    'BMI': 25.0,
    'DiabetesPedigreeFunction': 0.3,
    'Age': 28
}

# Make prediction
result = predictor.predict_single_patient(patient_data)

# Print results
print(f"Risk Assessment: {result['diabetes_risk']}")
print(f"Risk Probability: {result['risk_probability']}")
```

### Method 3: Multiple Patients

```python
import pandas as pd
from predict_diabetes import DiabetesPredictor

# Load multiple patients from CSV or create DataFrame
patients_df = pd.DataFrame([
    {'Pregnancies': 2, 'Glucose': 140, 'BloodPressure': 85, 'SkinThickness': 30,
     'Insulin': 0, 'BMI': 32.0, 'DiabetesPedigreeFunction': 0.4, 'Age': 45},
    {'Pregnancies': 0, 'Glucose': 95, 'BloodPressure': 70, 'SkinThickness': 20,
     'Insulin': 0, 'BMI': 22.0, 'DiabetesPedigreeFunction': 0.2, 'Age': 25}
])

predictor = DiabetesPredictor()
results = predictor.predict(patients_df)

# Access consensus predictions
consensus_predictions = results['consensus']['prediction']
consensus_probabilities = results['consensus']['probability']
```

## Understanding the Results

### Risk Assessment
- **HIGH**: Patient likely has diabetes (probability > 50%)
- **LOW**: Patient unlikely has diabetes (probability ≤ 50%)

### Probability Score
- Range: 0.000 to 1.000
- Higher values = Higher diabetes risk
- Example: 0.823 = 82.3% chance of diabetes

### Individual Model Predictions
The system uses 8 different machine learning models:
- **XGBoost**: Best overall performance (ROC-AUC: 0.8156)
- **Logistic Regression**: Good baseline model
- **Random Forest**: Tree-based ensemble method
- **SVM**: Support Vector Machine (with/without SMOTE)
- **k-NN**: k-Nearest Neighbors (with/without SMOTE)

### Consensus Prediction
The final prediction averages results from all models for more reliable predictions.

## Required Patient Data

All 8 features are required:

| Feature | Description | Typical Range | Notes |
|---------|-------------|---------------|-------|
| `Pregnancies` | Number of pregnancies | 0-17 | Integer |
| `Glucose` | Plasma glucose (mg/dL) | 44-199 | Use 0 if missing |
| `BloodPressure` | Diastolic BP (mmHg) | 24-122 | Use 0 if missing |
| `SkinThickness` | Triceps skin fold (mm) | 7-99 | Use 0 if missing |
| `Insulin` | 2-Hour serum insulin | 14-846 | Use 0 if missing |
| `BMI` | Body Mass Index | 18.2-67.1 | Use 0 if missing |
| `DiabetesPedigreeFunction` | Diabetes pedigree function | 0.078-2.42 | Family history score |
| `Age` | Age in years | 21-81 | Integer |

**Missing Values**: Use `0` for missing `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, or `BMI`. The system will automatically impute these values using advanced techniques.

## Model Performance

Current model performance on test data:

| Model | ROC-AUC Score |
|-------|---------------|
| XGBoost | 0.8156 |
| Logistic Regression | 0.8137 |
| Random Forest | 0.8119 |
| SVM + SMOTE | 0.8089 |
| Logistic Regression + SMOTE | 0.8081 |
| SVM | 0.7972 |
| k-NN | 0.7787 |
| k-NN + SMOTE | 0.7744 |

ROC-AUC scores range from 0-1, with 1.0 being perfect prediction. Scores above 0.8 are considered very good for medical prediction tasks.

## Visualizations

The system now automatically generates detailed visualizations after each prediction!

### Available Visualizations

1. **Comprehensive Diabetes Report** (`comprehensive_diabetes_report.png`)
   - Complete patient summary with all visualizations
   - Patient information table
   - Risk gauge showing probability
   - Model performance comparison
   - Individual model predictions
   - Feature importance analysis

2. **Model Performance Comparison** (`model_performance_comparison.png`)
   - Bar chart comparing ROC-AUC scores of all 8 models
   - Color-coded performance (green = good, red = poor)
   - Performance thresholds marked

3. **Patient Predictions** (`patient_predictions.png`)
   - Individual model predictions for the current patient
   - Risk probability gauge
   - Visual comparison of model consensus

4. **Feature Importance** (`feature_importance.png`)
   - Shows which patient features are most important for predictions
   - Separate charts for Random Forest and XGBoost models

### Using Visualizations

**Automatic Generation**:
```bash
# Predictions with automatic visualizations
./diabetes_env/bin/python test_prediction.py
```

**Standalone Visualization Generator**:
```bash
# Generate comprehensive visualization suite
./diabetes_env/bin/python generate_visualizations.py
```

**Customized Visualizations**:
```python
from predict_diabetes import DiabetesPredictor
from visualizer import DiabetesVisualizer

predictor = DiabetesPredictor()
visualizer = DiabetesVisualizer()

# Your patient data
patient_data = {...}
result = predictor.predict_single_patient(patient_data)

# Generate specific visualizations
visualizer.plot_patient_predictions(result, patient_data, save=True, show=True)
```

### Visualization Files Location

All charts are saved in the `graphs/` directory:
- 📊 `comprehensive_diabetes_report.png` - Complete analysis
- 📈 `model_performance_comparison.png` - Model comparison
- 🎯 `patient_predictions.png` - Individual predictions
- 🔍 `feature_importance.png` - Feature analysis

### Reading the Visualizations

**Risk Gauge**: Semicircular gauge showing patient's diabetes risk
- Green: Low Risk (0-30%)
- Yellow: Moderate Risk (30-50%)
- Orange: High Risk (50-70%)
- Red: Very High Risk (70-100%)

**Model Predictions Bar Chart**: Shows how each model voted
- Green bars: "No Diabetes" prediction
- Red bars: "Diabetes" prediction
- Height = confidence probability

**Feature Importance**: Which measurements matter most
- Higher bars = more important for prediction
- Helps understand what drives the risk assessment

## Troubleshooting

### "Models not found" Error
Run the training script first:
```bash
./diabetes_env/bin/python train_models.py
```

### Missing Dependencies
Make sure you've installed all required packages:
```bash
pip install -r requirements.txt
```

### "Module not found" Error
Make sure you're using the virtual environment Python:
```bash
./diabetes_env/bin/python your_script.py
```

### Feature Name Warnings
These warnings are harmless and don't affect predictions. They occur because sklearn expects consistent feature naming.

## Example Output

```
=== DIABETES PREDICTION EXAMPLE ===
Patient Data: {'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50}

🔍 PREDICTION RESULTS:
Overall Risk Assessment: HIGH
Risk Probability: 0.823

📊 Individual Model Predictions:
  xgboost: Diabetes (prob: 0.868)
  random_forest: Diabetes (prob: 0.873)
  logistic_regression: Diabetes (prob: 0.708)
  svm: Diabetes (prob: 0.798)
  ...
```

This indicates the 50-year-old patient has an 82.3% chance of having diabetes based on the consensus of all models.

## Important Notes

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Always consult healthcare professionals for medical decisions.

📊 **Data Quality**: The system handles missing values, but more complete data leads to better predictions.

🎯 **Model Updates**: Retrain models periodically with new data to maintain accuracy.

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Ensure you've run the training script
3. Verify patient data has all 8 required features
4. Review this guide for proper usage patterns

For questions about the underlying machine learning approach, see the original `diabetes_ml.ipynb` notebook.