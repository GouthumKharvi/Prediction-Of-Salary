# Prediction-Of-Salary
This project focuses on predicting employee salaries using machine learning techniques. It provides an end-to-end pipeline—from exploratory data analysis to model training and real-time prediction—allowing users to compare model performance and forecast salaries interactively through a Streamlit interface.

Perfect, Goutham 🙌 — since you now have a fully upgraded **Salary Prediction · Next-Gen** Streamlit app, here’s your **complete and professional README.md** (Markdown format).

It’s structured exactly like a real GitHub project — from **introduction to deployment**, including features, setup, file structure, data explanation, models used, and visuals section placeholders.

---

## 🧠 Salary Prediction · Next-Gen

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

**Salary Prediction · Next-Gen** is a **data analytics and machine learning project** that predicts employee salaries based on demographic, educational, and professional attributes.

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
| Web Framework    | Streamlit                      |
| UI Enhancements  | Custom CSS, HTML, Google Fonts |

---

## 📁 Project Structure

```
SalaryPredictionApp/
│
├── salarypredictionapp.py      # Main Streamlit app script
├── README.md                      # Documentation
├── requirements.txt               # Required Python packages
├── data/
│   └── Salary_Data_Based_country_and_race
├── models/                        # (Optional) Folder for saved models
   ├── best_model,scaler,encoders.joblib
   
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

# Save
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(mappings, 'models/encoders.pkl')

# Load
best_model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
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
streamlit run salarypredictionapp_v2.py
```

Then open the provided URL in your browser, usually:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📸 Screenshots

*(You can add actual images from your dashboard here)*

| Dashboard                                    | Model Comparison                                | Salary Prediction                              |
| -------------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| ![Dashboard](screenshots/dashboard_home.png) | ![Comparison](screenshots/model_comparison.png) | ![Prediction](screenshots/prediction_form.png) |

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
💼 Data Scientist | ML Engineer | Streamlit Developer
📧 [Contact via LinkedIn](https://www.linkedin.com) *(add your actual profile link)*

---

> 🏁 *This project is a showcase of end-to-end machine learning implementation — from raw data to deployed interactive analytics. Built to demonstrate model interpretability, data engineering, and interactive visualization expertise.*

---
