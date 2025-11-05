This project focuses on predicting employee salaries using machine learning techniques. It provides an end-to-end pipeline—from exploratory data analysis to model training and real-time prediction—allowing users to compare model performance and forecast salaries interactively through a Streamlit interface.


https://github.com/user-attachments/assets/346adb31-c9a6-42e4-90e8-b53221b7e213



---

## 🧠 Salary Prediction 

### *An Interactive Machine Learning Dashboard to Predict Employee Salaries*

💼 Built with **Python**, **Streamlit**, **scikit-learn**, and **XGBoost**

---

### 🗂️ Table of Contents

* [📘 Overview](#-overview)
* [🎯 Objectives](#-objectives)
* [⚙️ Features](#️-features)
* [🧩 Technologies Used](#-technologies-used)
* [📁 Project Structure](#-project-structure)
* [🧹 Data Preprocessing & Cleaning](#-data-preprocessing--cleaning)
* [🧠 Machine Learning Models](#-machine-learning-models)
* [📊 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [🚀 Streamlit Web Application](#-streamlit-web-application)
* [🔍 Model Evaluation Metrics](#-model-evaluation-metrics)
* [💾 Saving & Loading Models](#-saving--loading-models)
* [📈 Prediction Workflow](#-prediction-workflow)
* [📦 Installation & Setup](#-installation--setup)
* [🌐 Running the App](#-running-the-app)
* [📸 Screenshots](#-screenshots)
* [🚧 Future Enhancements](#-future-enhancements)
* [👨‍💻 Author](#-author)

---

## 📘 Overview

**Salary Prediction ** is a **data analytics and machine learning project** that predicts employee salaries based on demographic, educational, and professional attributes.

It provides:

* An **interactive dashboard** built in **Streamlit**
* Full **EDA and visualization** tools
* **Model comparison** between multiple ML algorithms
* Real-time **salary prediction form**
* A sleek, **modern dark/light UI**

This project demonstrates a complete data science workflow — from **data cleaning** and **feature engineering** to **model development**, **evaluation**, and **deployment** through a web app interface.

---

## 🎯 Objectives

The primary goal of this project is to:

1. Analyze employee salary data to identify key influencing features.
2. Build and compare different machine learning models for salary prediction.
3. Provide a real-time, interactive dashboard for salary estimation.
4. Present a visually appealing and professional app suitable for portfolio or production deployment.

---

## ⚙️ Features

✅ **Automatic Data Cleaning**

* Removes irrelevant or high missing-value columns (>30%)
* Handles missing or non-numeric salary values automatically

✅ **Feature Engineering**

* Adds new columns:

  * `Experience_Level` (Entry, Junior, Mid, Senior, Lead)
  * `Is_Senior` flag
  * `Role_Group` (job category normalization)
* Standardized scaling with `StandardScaler`

✅ **Exploratory Data Analysis (EDA)**

* Interactive plots for salary distribution and relationships
* Correlation heatmap
* Descriptive statistics and KPIs

✅ **Modeling & Comparison**

* Six ML models:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * XGBoost Regressor
  * Support Vector Regressor (SVR)
  * K-Nearest Neighbors (KNN)
* Auto-training and performance comparison (R², MAE, RMSE)
* Top-3 models highlighted with metrics

* Two DL Models:
  
  * Wide Neaural Network
  * Deep Neural Network

✅ **Salary Prediction Interface**

* User inputs demographic and professional info
* Selects model for prediction
* Real-time prediction with encoded feature mapping
* Styled result card showing the estimated salary

✅ **Cutting-Edge UI**

* Theme toggle (🌙 Dark / ☀️ Light)
* Glassmorphism + Neon gradient design
* Responsive cards, gradient headers, and hover animations

---

## 🧩 Technologies Used

| Category         | Libraries / Tools              |
| ---------------- | ------------------------------ |
| Programming      | Python 3.11                    |
| Data Handling    | pandas, numpy                  |
| Visualization    | matplotlib, seaborn            |
| Machine Learning | scikit-learn, xgboost          |
| Deep Learning    | Keras, Pytorch                 |
| Web Framework    | Streamlit                      |
| UI Enhancements  | Custom CSS, HTML, Google Fonts |

---

## 📁 Project Structure

```
SalaryPredictionApp/
│
├── salarypredictionapp.py               
├── README.md                            
├── requirements.txt                     
│
├── data/
│   └── Salary_Data_Based_country_and_race.csv     
│
├── models/                              
│   ├── best_model,scaler,encoders,metadata.joblib               


│
├── model_comparison/                    
│   ├── decision_tree_hyperparameter_tuned_model_results.csv
│   ├── knn_hyperparameter_tuned_model_results.csv
│   ├── knn_tuned_model_results.csv
│   ├── linear_model_results.csv
│   ├── mlp_hyperparameter_tuned_model_results.csv
│   ├── model_results.csv
│   ├── random_forest_tuned_model_results.csv
│   ├── Ridge_regression_hyperparameter_tuned_model_results.csv
│   ├── svr_hyperparameter_tuned_model_results.csv
│   └── xgboost_tuned_model_results.csv
│
└── screenshots/                        
    ├── dashboard_home.png
    ├── model_comparison.png
    └── prediction_form.png

```

---

## 🧹 Data Preprocessing & Cleaning

1. **Load dataset** from CSV file using pandas.
2. Drop the redundant column `Unnamed: 0` (auto-index).
3. Remove any columns with **more than 30% missing values**.
4. Drop rows with missing or non-numeric salary values.
5. Create new engineered features:

   * **Years_of_Experience**
   * **Experience_Level (bins)**
   * **Is_Senior (binary flag)**
   * **Role_Group (derived from job title)**
6. Encode categorical features using **custom mapping dictionaries** to preserve label meaning.
7. Apply **StandardScaler** for normalization before training.

---

## 🧠 Machine Learning Models

| Model                              | Description                                             |
| ---------------------------------- | ------------------------------------------------------- |
| **Linear Regression**              | Baseline model for salary prediction.                   |
| **Decision Tree**                  | Non-linear decision-based approach.                     |
| **Random Forest**                  | Ensemble of trees improving stability and accuracy.     |
| **XGBoost**                        | Gradient boosting algorithm for strong performance.     |
| **SVR (Support Vector Regressor)** | Fits optimal regression hyperplane.                     |
| **KNN (K-Nearest Neighbors)**      | Predicts salary by similarity among nearby data points. |
| **Wide Neaural Network**           | A Wide Neural Network is used to capture complex feature interactions in tabular or structured data by having few layers with many neurons in each layer.|
| **Deep Neaural Network**           | A Deep Neural Network is used to learn hierarchical and complex patterns from data through many hidden layers stacked sequentially.|

All models are trained on the same feature set, and performance is compared using R², MAE, and RMSE.

---

## 📊 Exploratory Data Analysis (EDA)

EDA is available interactively through the dashboard:

* **Salary Distribution**: Histogram + KDE curve
* **Age vs Salary Scatter Plot** (by gender)
* **Correlation Heatmap** for numeric relationships
* **Experience Level Breakdown**

Each plot dynamically updates using **Matplotlib** and **Seaborn**, styled in theme colors.

---

## 🚀 Streamlit Web Application

The app has four major sections:

| Section                      | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| **📂 Overview**              | Shows dataset summary, missing columns removed, and feature stats. |
| **📊 Exploratory Data**      | Displays visual insights and summary metrics.                      |
| **🤖 Modeling & Comparison** | Trains all models and compares them side-by-side.                  |
| **📈 Salary Prediction**     | Real-time salary prediction based on user inputs.                  |

---

## 🔍 Model Evaluation Metrics

For each model, the following metrics are calculated:

* **R² Score** (Coefficient of determination)
* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)

A bar chart compares R² values across models, highlighting top performers.

---

## 💾 Saving & Loading Models

> (Optional) You can persist trained models for deployment.
> Example:

```python
import joblib
import os
from datetime import datetime

# Define save path
save_path = r'C:\Users\Gouthum\Downloads\inlighn projects(practical)\Model comparision'
os.makedirs(save_path, exist_ok=True)

# Save the best Random Forest model
model_filename = 'best_random_forest_model.joblib'
model_filepath = os.path.join(save_path, model_filename)

# Save the tuned Random Forest model
joblib.dump(rf_model_tuned, model_filepath)

print(f"Best model saved to: {model_filepath}")
print(f"Model type: Random Forest (Tuned)")
print(f"Performance: 95.00% R²")
print(f"File size: {os.path.getsize(model_filepath) / 1024:.2f} KB")

# Also save the preprocessing components for complete pipeline
scaler_filepath = os.path.join(save_path, 'salary_scaler.joblib')
label_encoders_filepath = os.path.join(save_path, 'salary_label_encoders.joblib')

# Save scaler
joblib.dump(scaler, scaler_filepath)

# Save label encoders (if you have them stored)
# joblib.dump(label_encoders_dict, label_encoders_filepath)

print(f"\nPreprocessing components saved:")
print(f"Scaler: {scaler_filepath}")
# print(f"Label encoders: {label_encoders_filepath}")

# Create model metadata
metadata = {
    'model_name': 'Random Forest (Tuned)',
    'model_type': 'RandomForestRegressor',
    'performance': {
        'r2_score': 0.9500,
        'rmse': 0.2261,
        'mae': 0.1175
    },
    'features': ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Country', 'Race'],
    'target': 'Salary',
    'preprocessing': 'StandardScaler + LabelEncoder',
    'hyperparameters': 'Tuned via RandomizedSearchCV',
    'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_shape': {
        'train_samples': 5363,
        'test_samples': 1341,
        'features': 8
    }
}

# Save metadata
metadata_filepath = os.path.join(save_path, 'best_model_metadata.joblib')
joblib.dump(metadata, metadata_filepath)

print(f"Model metadata: {metadata_filepath}")

**Load and Use Saved Model**
def load_best_model(model_path):
    """Load the saved Random Forest model"""
    model = joblib.load(model_path)
    return model

def load_metadata(metadata_path):
    """Load model metadata"""
    metadata = joblib.load(metadata_path)
    return metadata

# Load model for predictions
loaded_model = load_best_model(model_filepath)
loaded_metadata = load_metadata(metadata_filepath)

print("MODEL SUCCESSFULLY LOADED:")
print(f"Model: {loaded_metadata['model_name']}")
print(f"Performance: {loaded_metadata['performance']['r2_score']:.4f} R²")

# Test prediction on first few samples
sample_predictions = loaded_model.predict(X_test.head(3))
print(f"\nSample predictions: {sample_predictions}")

```

---

## 📈 Prediction Workflow

1. User enters **profile information** in the app (Age, Gender, Education, Job Title, etc.)
2. App encodes categorical data using stored mappings.
3. Scaler standardizes numeric inputs.
4. The chosen trained model predicts salary.
5. Result is displayed in a styled success card.

---

## 📦 Installation & Setup

### 🔹 Prerequisites

* Python 3.10 or above
* pip or conda environment
* Git (optional)

### 🔹 Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SalaryPredictionApp.git
cd SalaryPredictionApp

# 2. (Optional) Create and activate a virtual environment
conda create -n salaryapp python=3.11 -y
conda activate salaryapp

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 🌐 Running the App

```bash
# Navigate to the project folder
cd "C:\Users\Gouthum\Downloads\inlighn projects(practical)"

# Run the Streamlit app
streamlit run salarypredictionapp.py
```

Then open the provided URL in your browser, usually:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📸 Screenshots

| -------------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
<img width="1916" height="851" alt="Model Training Evaulation" src="https://github.com/user-attachments/assets/cdbb5929-4284-4a86-b49a-82064009c55a" />
<img width="1916" height="851" alt="Model Training Evaulation" src="https://github.com/user-attachments/assets/bd515319-f3a6-4c80-8373-99ea806307db" />
<img width="1909" height="916" alt="EDA" src="https://github.com/user-attachments/assets/4868e38b-b4bb-445c-b87d-41fe29f05d05" />
<img width="1914" height="878" alt="Dashboard" src="https://github.com/user-attachments/assets/75224060-a6ba-43f9-b4fa-9b4ad434e773" />
<img width="1096" height="423" alt="ccompare model" src="https://github.com/user-attachments/assets/94eaab19-2183-4bff-8ff1-2ebf02467df4" />


---

## 🚧 Future Enhancements

🔮 Planned upgrades for version 3.0:

* ✅ Save & load best models automatically
* ✅ Add SHAP explainability for feature importance
* ✅ Add global filter panel for data selection
* ✅ Integrate database (PostgreSQL / Snowflake)
* ✅ Add authentication for user sessions
* ✅ Deploy to Streamlit Cloud / AWS EC2

---

## 👨‍💻 Author

**Goutham Kharvi**
📍 *Bengaluru, Karnataka*
💼 Data Scientist |AI/ ML Engineer | Streamlit Developer
📧 [Contact via LinkedIn]( https://www.linkedin.com/in/gouthum-kharvi-2366a6219/)*

---

> 🏁 *This project is a showcase of end-to-end machine learning implementation — from raw data to deployed interactive analytics. Built to demonstrate model interpretability, data engineering, and interactive visualization expertise.*

---
