# Project Car Price Prediction on DVM-Car dataset

## Introduction

Welcome to the **Car Price Prediction** project! The goal is to predict car prices as accurately as possible using machine learning models. These models are trained on the renowned **DVM-Car Dataset**, which comprises over 250,000+ car samples and advertisements.

This repository contains the core components of the project, including:
- **Machine Learning Models (3):** Linear Regression, Random Forest Regressor, and XGBoost.
- **Exploratory Data Analysis (EDA) file.**
- **Preprocessing Pipeline**

## Deployment (may takes a while to load)
- **RESTful API**: [car-price-prediction-dvm.onrender.com](https://car-price-prediction-dvm.onrender.com/) \
(add **/docs** to see API document)


## Features

* **Exploratory Data Analysis (EDA):** Provides charts and statistics offering deep insights into the car dataset.
* **Machine Learning Models:** Utilizes various ML models to train and build accurate car price prediction models.
* **Prediction API:** A RESTful API built with FastAPI for integrating car price predictions into other applications.
* **Web Interface:** A simple web application allowing users to input car information and receive instant price predictions.
* **Containerization (Docker):** Ensures easy deployment across different platforms.

## Project Structure

Describe your project's directory structure to help readers quickly locate important files.
```
.
├── .venv/                      # Python virtual environment
├── app/                        # Directory containing the main API application
│   ├── pycache/
│   └── api.py                  # FastAPI API endpoints
├── notebooks/                  # Jupyter Notebooks for EDA and model experimentation
│   ├── eda.ipynb               # Main EDA file
│   ├── main-draft.ipynb        # A draft containing notes about the project and pipeline
│   └── preprocessing-and-model-experiment.ipynb # Main notebook containing the project's preprocessing pipeline and model experiments.
├── src/                        # Source code for the chosen model and utilities
│   ├── pycache/
│   ├── .gradio/                # Contains Gradio-related configurations or files
│   ├── best-model-pipeline.ipynb # Notebook with the optimized model pipeline
│   ├── data_preprocessor.py    # Data preprocessor class
│   ├── model.pkl               # Trained machine learning model
│   ├── preprocessor.pkl        # Saved preprocessor object
│   └── prototype.ipynb         # Gradio prototype for model testing
├── .gitignore
├── car-dataset.csv             # The car dataset used for training
├── Dockerfile                  # Docker configuration file
└── requirements.txt            # List of required Python libraries
```
## Tech Stack

List the main languages, frameworks, and libraries.

* **Language:** Python
* **API Framework:** FastAPI
* **ML Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, category_encoders,...
* **Deployment:** Docker (on Render.com), Uvicorn
* **Web Interface:** HTML, CSS, Javascript
