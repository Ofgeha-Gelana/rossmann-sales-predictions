# Rossmann Sales Prediction

This repository contains an end-to-end solution for forecasting daily sales at Rossmann Pharmaceuticals stores across various cities. The project leverages machine learning and deep learning techniques to provide six-week-ahead sales predictions, helping the finance team make data-driven decisions.

---

## **Project Overview**
Rossmann Pharmaceuticals aims to forecast sales for its stores while considering factors like promotions, competition, holidays, seasonality, and locality. The goal is to empower the finance team with accurate sales forecasts, enabling strategic planning and operational efficiency.

---

## **Key Features**
- **Exploratory Data Analysis (EDA):**
  - Visualize sales trends before, during, and after holidays.
  - Investigate the impact of promotions, holidays, and competition on sales.
  - Explore customer behavior trends and store-specific characteristics.

- **Machine Learning & Deep Learning Models:**
  - Predict daily sales for six weeks into the future.
  - Evaluate and compare different algorithms for optimal performance.

- **Feature Engineering:**
  - Develop features such as seasonality indicators and lagged sales.
  - Handle missing data and outliers to improve prediction accuracy.

- **Web Interface for Predictions:**
  - Serve forecasts on a user-friendly web interface for finance analysts.

- **Logging and Reproducibility:**
  - Implement robust logging using Python's `logging` library for traceability.

---

## **Data**
The project uses two main datasets:
1. **Training Data:**
   - Contains historical daily sales, customer counts, and promotional details.
2. **Store Data:**
   - Includes store metadata such as store type, assortment levels, and competitor information.

### **Key Columns**
- `Sales`: Daily turnover (target variable).
- `Customers`: Number of customers visiting the store.
- `Promo`: Indicator for promotional campaigns.
- `StateHoliday`: Information on public and school holidays.
- `CompetitionDistance`: Distance to the nearest competitor.

---

## **Project Structure**
```plaintext
rossmann-sales-predictions/
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── modeling.ipynb      # Model training and evaluation
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model.py         # Machine learning and deep learning models
│   ├── web_interface.py       # Code for serving predictions
├── logs/                  # Log files for reproducibility
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files and folders
