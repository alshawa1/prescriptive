# Telecom Churn Prediction & Recommendation System

This project analyzes telecom customer data to predict churn and recommends retention strategies using an AI-powered K-Means clustering system.

## 📂 Project Structure
- **`churn_app.py`**: The main Streamlit application.
- **`telecom_churn.csv`**: The dataset used for analysis and training.
- **`requirements.txt`**: List of dependencies.
- **`telecom_churn_analysis.ipynb`**: Jupyter Notebook with detailed exploratory and predictive analysis.

## 🚀 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit App:
   ```bash
   streamlit run churn_app.py
   ```

## ☁️ How to Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select the repository.
4. Select `churn_app.py` as the main file.
5. Click **Deploy**.

## 🧠 Features
- **Churn Prediction**: Random Forest model to predict if a customer will leave.
- **Smart Recommendations**: K-Means clustering segments customers (e.g., "High Value", "Heavy Data User") and suggests tailored retention strategies.
- **ROI Calculator**: Estimates the net financial benefit of the recommended strategy.
