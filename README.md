# diabetes-ml-project
Machine Learning project for CS512 Machine Learning in Medicine and Health 

Project Overview

This project investigates the use of machine learning techniques for predicting diabetes based on clinical and demographic data. The goal is to evaluate how advanced preprocessing, class imbalance handling, and different learning algorithms affect predictive performance in a medical classification task.

The project focuses on realistic challenges commonly encountered in healthcare data, such as missing values, physiologically implausible measurements, and class imbalance.

📊 Dataset

Name: Pima Indians Diabetes Dataset

Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

Accessed via: Kaggle

Samples: 768

Features: 8 clinical predictors + 1 binary target

The dataset contains medical measurements such as glucose level, body mass index, insulin, and age, with the target variable indicating diabetes status.

📎 Dataset link:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

🧪 Methodology
1. Exploratory Data Analysis (EDA)

Analysis of feature distributions

Identification of class imbalance

Visualization of relationships between features and diabetes outcome

2. Data Preprocessing

Treatment of physiologically implausible zero values as missing

Advanced multivariate imputation (KNN and Iterative Imputation)

Feature scaling using standardization

Class imbalance handling using SMOTE (applied only to training data)

3. Machine Learning Models

The following models were implemented and evaluated:

Logistic Regression

k-Nearest Neighbors (k-NN)

Support Vector Machine (SVM)

Random Forest

Gradient Boosted Decision Trees (XGBoost)

4. Evaluation Metrics

Models were evaluated on an unseen test set using:

Accuracy

Balanced Accuracy

Sensitivity (Recall)

Specificity

F1-score

ROC-AUC

Sensitivity was emphasized due to its importance in medical screening tasks.

🏆 Results

Ensemble models outperformed classical approaches.

Class imbalance handling significantly improved recall.

XGBoost achieved the best overall performance, with:

Recall ≈ 0.69

ROC-AUC ≈ 0.82

Feature importance analysis identified glucose, BMI, age, and insulin as the most influential predictors.


How to Run the Notebook

Clone the repository:

git clone https://github.com/anespiki/diabetes-ml-project.git
cd diabetes-ml-project

Install required dependencies:

pip install -r requirements.txt


Launch Jupyter Notebook:

jupyter notebook


Open the file diabetes_ml.ipynb and run all cells sequentially.
