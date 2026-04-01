# 🌾 Local Agricultural Price & Climate Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightgrey.svg)](https://pandas.pydata.org/)

## 📌 Overview
This project is an end-to-end Machine Learning pipeline designed to predict the session-to-session percentage change in wholesale agricultural prices. By leveraging 5 years of historical market data and combining it with lagged meteorological data (Open-Meteo API), this project aims to uncover the non-linear relationship between climate shocks and agricultural supply chains.

Developed as a comprehensive data science initiative at **Konya Food and Agriculture University**, this project demonstrates advanced ETL processes, inflation-resistant feature engineering, and rigorous model evaluation.

## 🚀 Key Methodologies & Feature Engineering
* **Inflation-Proof Target Variable:** Analyzing nominal wholesale prices in a high-inflation economy results in non-stationary data. To neutralize macroeconomic inflation, the target variable was transformed from *Absolute Price (TL)* to **Daily Percentage Price Change (%)**.
* **Climate Lags:** Weather doesn't impact crop prices instantly. The pipeline engineers 7-day, 14-day, and 30-day rolling averages for temperature and precipitation to capture the delayed biological impact of weather (e.g., frost or drought) on harvests.
* **Zero-Inflated Data Handling:** The EDA revealed a highly zero-inflated market behavior (prices remain unchanged ~75% of the time). 

## 📊 Machine Learning Modeling & Results
Due to the time-series nature of the data, a chronological split (80% Train, 20% Test) was used to prevent data leakage. Three algorithms were compared to predict the percentage price volatility:

1. **Linear Regression (Baseline)**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

### 🏆 The "Occam's Razor" Reality
Surprisingly, the baseline **Linear Regression** model outperformed the complex ensemble methods, achieving a Mean Absolute Error (MAE) of **~4.58%**. 

**Why?** Tree-based models like XGBoost attempted to aggressively find complex patterns to predict rare, extreme price spikes within the zero-inflated data, leading to overfitting. The Linear Regression model took a more conservative approach, yielding a lower overall error across the timeline.

### 🔍 Feature Importance
The linear coefficients revealed that the **specific crop type** (e.g., *Kıl Biber*, *Dere Otu*) is the strongest baseline predictor of volatility. Among the engineered climate factors, **30-Day Average Rainfall** and **Daily Average Temperature** emerged as the top continuous predictors, mathematically validating the long-term impact of drought and short-term temperature shocks on market supply.

## 🛠️ Technologies Used
* **Data Engineering:** Python, Pandas, NumPy, BeautifulSoup, Requests (API)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Visualization:** Matplotlib, Seaborn

## 📂 Repository Structure
* `feature_engineering.py`: The ETL pipeline that cleans raw text data, handles formatting anomalies, engineers the percentage target variable, and merges climate data.
* `Local_Agricultural_Price_and_Climate_Predictor.ipynb`: The main Google Colab notebook containing EDA, statistical distribution analysis, and the machine learning model comparison.
* `konya_tarim_final_engineered_dataset.csv`: The final, cleaned, and ML-ready dataset.

## 💡 Business Impact
Predicting daily agricultural price fluctuations with an error margin of ~4.5% provides actionable economic value. It acts as an early-warning radar for local wholesalers and traders to adjust procurement strategies before climate-induced supply shocks cause price surges.
