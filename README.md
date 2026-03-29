# House Price Prediction Pipeline

## Overview

This project builds a machine learning model to predict house prices using the Ames Housing dataset. It demonstrates a clean, end-to-end workflow including preprocessing, feature engineering, model training, and prediction.

## Problem

Predict the `SalePrice` of a house based on its features.

## Approach

* Performed exploratory data analysis (EDA)
* Handled missing values and categorical data using pipelines
* Created meaningful features:
  * HouseAge
  * RemodAge
  * TotalBath
  * TotalSF
* Trained models:
  * Linear Regression (baseline)
  * Random Forest
  * XGBoost (final model)

## Final Model

* Model: XGBoost Regressor
* Integrated with preprocessing using a pipeline
* Trained on full dataset
* Saved using joblib

## Key Learnings

* Avoiding data leakage
* Importance of feature engineering
* Using pipelines for consistency
* Building reproducible ML workflows

## How to Run

1. Train model
2. Save pipeline (`.pkl` file)
3. Load test data
4. Apply same feature engineering
5. Generate predictions
6. Create submission CSV

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost

## Result

Achieved a working Kaggle submission with a solid baseline performance and scope for further improvement through tuning and ensembling.

## Future Work

* Hyperparameter tuning
* Cross-validation
* Model ensembling
