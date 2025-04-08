Time Series Forecasting of Stock Prices Based on Fundamental Financial Indicators

## ECE9309 - Final Project

This repository contains code for a stock price prediction system that leverages machine learning models and financial indicators to forecast Apple (AAPL) stock prices.

## Project Overview

This project explores different approaches to predicting stock prices by combining:

- Historical stock price data
- Financial statement metrics and ratios
- Various machine learning and statistical models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SVR (Support Vector Regression)
  - LSTM (Long Short-Term Memory neural networks)

The models are evaluated using performance metrics such as MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R² to determine their effectiveness in predicting future stock price movements.

## Repository Contents

- **Stock_Price_Prediction_with_Financial_Indicators.ipynb**: Main notebook containing data analysis, model implementations, and result visualizations
- **Alpha_Vantage_AAPL_Financial_Data.ipynb**: Notebook for retrieving and processing financial statement data from Alpha Vantage API

## Data Sources

- Historical daily stock price data for Apple Inc. (AAPL)
- Quarterly financial reports (income statements, balance sheets) from Alpha Vantage API
- Derived financial ratios and indicators including:
  - Net Profit Margin
  - Return on Assets (ROA)
  - EBIT Margin
  - P/E and P/B Ratios
  - Operating Margin
  - Gross Margin
  - R&D to Revenue
  - Cost Efficiency
  - Operating Leverage

## Methodology

### Data Processing

1. Retrieval of stock prices and financial statement data
2. Creation of financial ratios and indicators
3. Data cleaning and normalization
4. Feature selection using correlation analysis and mutual information

### Models Implemented

1. **ARIMA (AutoRegressive Integrated Moving Average)**

   - Time series analysis with differencing to achieve stationarity
   - Parameter selection via ACF and PACF analysis

2. **SVR (Support Vector Regression)**

   - Univariate prediction using only historical prices
   - Multivariate prediction incorporating financial indicators
   - Hyperparameter tuning using grid search

3. **LSTM (Long Short-Term Memory Neural Network)**
   - Sequential model capturing temporal dependencies
   - Single feature and multi-feature implementations
   - Custom architecture with weight initialization techniques

## Results

Models are compared based on:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score (coefficient of determination)

Visualizations include:

- Actual vs. predicted price charts
- Feature importance analysis
- Correlation heatmaps of financial indicators

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- pmdarima
- torch (PyTorch)
- tensorflow/keras

## Usage

1. Run Alpha_Vantage_AAPL_Financial_Data.ipynb to collect and process financial data (requires API key)
2. Run Stock_Price_Prediction_with_Financial_Indicators.ipynb to perform analysis and train models

## Acknowledgments

- Alpha Vantage API for providing financial statement data
- ECE9309 course instructors and classmates for feedback and support
