import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# 🎨 Advanced Modern CSS Styling
# ------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dark Mode Base */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        /* Animated Gradient Header */
        .main-title {
            font-size: 52px;
            font-weight: 800;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #667eea 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 4s ease infinite;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }
        
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .subtitle {
            text-align: center;
            color: #a78bfa;
            font-size: 18px;
            font-weight: 400;
            margin-bottom: 40px;
            opacity: 0.9;
        }

        /* Glassmorphism Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(30, 27, 75, 0.7);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(167, 139, 250, 0.2);
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stSidebar"] * {
            color: #e9d5ff !important;
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            font-weight: 600;
            font-size: 16px;
            color: #f3e8ff !important;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(167, 139, 250, 0.1);
            padding: 12px 16px;
            border-radius: 12px;
            margin: 6px 0;
            transition: all 0.3s ease;
            border: 1px solid rgba(167, 139, 250, 0.2);
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(167, 139, 250, 0.2);
            transform: translateX(5px);
            border-color: rgba(167, 139, 250, 0.4);
        }

        /* Premium Card Design */
        .card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(167, 139, 250, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
            border-color: rgba(167, 139, 250, 0.4);
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetricLabel"] {
            color: #c4b5fd !important;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Modern Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px 32px;
            font-weight: 700;
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            width: 100%;
        }
        
        div.stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        div.stButton > button:active {
            transform: translateY(-1px);
        }

        /* Input Fields */
        .stSelectbox, .stNumberInput {
            background: rgba(30, 27, 75, 0.4);
            border-radius: 12px;
        }
        
        input, select {
            background: rgba(30, 27, 75, 0.6) !important;
            border: 1px solid rgba(167, 139, 250, 0.3) !important;
            border-radius: 10px !important;
            color: #e9d5ff !important;
            padding: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        input:focus, select:focus {
            border-color: rgba(167, 139, 250, 0.6) !important;
            box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1) !important;
        }

        /* DataFrames */
        [data-testid="stDataFrame"] {
            background: rgba(30, 27, 75, 0.4);
            border-radius: 15px;
            border: 1px solid rgba(167, 139, 250, 0.2);
            overflow: hidden;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #f3e8ff !important;
            font-weight: 700 !important;
        }
        
        p, span, div {
            color: #e9d5ff !important;
        }

        /* Success Box */
        .stSuccess {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }

        /* Plots */
        .stPlotlyChart, .stPyplot {
            background: rgba(30, 27, 75, 0.4);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(167, 139, 250, 0.2);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 27, 75, 0.4);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1 class='main-title'>💼 Salary Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict and analyze employee salaries using cutting-edge Machine Learning algorithms</p>", unsafe_allow_html=True)

# ------------------------------
# Sidebar Navigation
# ------------------------------
menu = st.sidebar.radio(
    "🧭 Navigation",
    ["📂 Data Overview", "📊 Exploratory Data Analysis", "🤖 Model Training & Evaluation", "📈 Salary Prediction"]
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Gouthum/Downloads/Salary_Data_Based_country_and_race.csv")

    # Clean Data
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'Salary' in df.columns:
        df = df.dropna(subset=['Salary'])

    return df

df = load_data()

# ------------------------------
# Encode Data
# ------------------------------
def encode_data(data):
    df_encoded = data.copy()
    if 'Unnamed: 0' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['Unnamed: 0'])
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    df_encoded = df_encoded.dropna()
    return df_encoded

# ------------------------------
# 1️⃣ Data Overview
# ------------------------------
if menu == "📂 Data Overview":
    st.markdown("### 📘 Data Overview")
    with st.container():
        st.dataframe(df.head(), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Features", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### 📄 Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes,
        "Missing Values": df.isnull().sum()
    })
    st.dataframe(info_df, use_container_width=True)

# ------------------------------
# 2️⃣ EDA
# ------------------------------
elif menu == "📊 Exploratory Data Analysis":
    st.markdown("### 📊 Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Salary Distribution")
        fig, ax = plt.subplots(facecolor='#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        sns.histplot(df['Salary'], kde=True, color="#667eea", ax=ax)
        ax.tick_params(colors='#e9d5ff')
        ax.spines['bottom'].set_color('#a78bfa')
        ax.spines['left'].set_color('#a78bfa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Salary', color='#e9d5ff')
        ax.set_ylabel('Frequency', color='#e9d5ff')
        st.pyplot(fig)
    with col2:
        st.markdown("#### Age vs Salary by Gender")
        fig, ax = plt.subplots(facecolor='#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        sns.scatterplot(data=df, x="Age", y="Salary", hue="Gender", palette="viridis", ax=ax, alpha=0.7)
        ax.tick_params(colors='#e9d5ff')
        ax.spines['bottom'].set_color('#a78bfa')
        ax.spines['left'].set_color('#a78bfa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Age', color='#e9d5ff')
        ax.set_ylabel('Salary', color='#e9d5ff')
        legend = ax.legend()
        legend.get_frame().set_facecolor('#1e1b4b')
        for text in legend.get_texts():
            text.set_color('#e9d5ff')
        st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    df_encoded = encode_data(df)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1b4b')
    ax.set_facecolor('#1e1b4b')
    sns.heatmap(df_encoded.corr(), annot=True, cmap="twilight", ax=ax, cbar_kws={'label': 'Correlation'})
    ax.tick_params(colors='#e9d5ff')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#e9d5ff')
    cbar.set_label('Correlation', color='#e9d5ff')
    st.pyplot(fig)

# ------------------------------
# 3️⃣ Model Training & Evaluation
# ------------------------------
elif menu == "🤖 Model Training & Evaluation":
    st.markdown("### 🤖 Train and Evaluate Models")

    df_encoded = encode_data(df)
    X = df_encoded.drop(columns=['Salary'])
    y = df_encoded['Salary']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox(
        "Select a Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "Support Vector Regressor", "KNN"]
    )

    if st.button("🚀 Train Model"):
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_choice == "XGBoost":
            model = XGBRegressor(random_state=42)
        elif model_choice == "Support Vector Regressor":
            model = SVR()
        elif model_choice == "KNN":
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.markdown("### 📈 Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots(facecolor='#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        sns.scatterplot(x=y_test, y=y_pred, color="#667eea", alpha=0.6, s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, alpha=0.7)
        ax.set_xlabel("Actual Salary", color='#e9d5ff', fontsize=12)
        ax.set_ylabel("Predicted Salary", color='#e9d5ff', fontsize=12)
        ax.set_title(f"Actual vs Predicted ({model_choice})", color='#f3e8ff', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#e9d5ff')
        ax.spines['bottom'].set_color('#a78bfa')
        ax.spines['left'].set_color('#a78bfa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#a78bfa')
        st.pyplot(fig)

# ------------------------------
# 4️⃣ Salary Prediction
# ------------------------------
elif menu == "📈 Salary Prediction":
    st.markdown("### 💰 Predict Employee Salary")

    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, step=1)
        gender = st.selectbox("Gender", df['Gender'].unique())
        education = st.selectbox("Education Level", df['Education Level'].unique())
        job = st.selectbox("Job Title", df['Job Title'].unique())
    
    with col2:
        exp = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
        country = st.selectbox("Country", df['Country'].unique())
        race = st.selectbox("Race", df['Race'].unique())
        model_choice = st.selectbox(
            "Select Model for Prediction",
            ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "Support Vector Regressor", "KNN"]
        )

    if st.button("🔮 Predict Salary"):
        df_encoded = encode_data(df)
        scaler = StandardScaler()
        X = df_encoded.drop(columns=['Salary'])
        y = df_encoded['Salary']
        X_scaled = scaler.fit_transform(X)

        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [job],
            "Years of Experience": [exp],
            "Country": [country],
            "Race": [race]
        })

        input_encoded = encode_data(input_df)
        input_scaled = scaler.transform(input_encoded)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_choice == "XGBoost":
            model = XGBRegressor(random_state=42)
        elif model_choice == "Support Vector Regressor":
            model = SVR()
        elif model_choice == "KNN":
            model = KNeighborsRegressor()

        model.fit(X_scaled, y)
        salary_pred = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Salary: **${salary_pred:,.2f}** 💵")