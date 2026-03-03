# 📊 Hybrid Sales Forecasting Web App

An end-to-end **Machine Learning web application** for retail sales forecasting using a **hybrid approach**.  
The model combines a Prophet-like trend & seasonality method with a **Gradient Boosting regressor**, deployed using **Streamlit**.

---

## 🚀 Live Application
🔗 https://hybrid-sales-forecasting-app-duugsiw9p2wfqtrq7pbkgj.streamlit.app/

---

## 📌 Problem Statement
Retail sales data is affected by trend, seasonality, and complex non-linear patterns.  
Single models often fail to capture all components effectively.

This project addresses the problem by combining:
- Trend & seasonality modeling
- Machine learning regression
- Hybrid ensemble forecasting

---

## 🧠 Methodology

### 1️⃣ Prophet-like Model
- Linear Regression for long-term trend
- Weekly seasonality extraction

### 2️⃣ Machine Learning Model
- Gradient Boosting Regressor
- Lag features and rolling averages

### 3️⃣ Hybrid Forecast
- Weighted combination of both models
- Improved stability and accuracy

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit



## 📊 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE
- Accuracy (%)

---

## 🎯 Features
✔ Upload CSV dataset  
✔ Automated feature engineering  
✔ Hybrid forecasting model  
✔ Interactive visualizations  
✔ Deployed web application  

👤 Author
Prawin Kanna Palaniyappan
Aspiring Data Analyst & Machine Learning Engineer
---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py


