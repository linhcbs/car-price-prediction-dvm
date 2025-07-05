# Project Car Price Prediction on DVM-Car dataset

## 🎯 Introduction

Welcome to project Car Price Prediction! Where the goal is to predict car prices as accurate as possible using machine learning models which are trained on the famous DVM-Car Dataset with over 250 000+ car samples and advertisements.

This repository contains core components of the project, including: 
- ML models (3): Linear Regression, Random Forest Regressor, XGBoost
- EDA file (Exploratory Data Analysis) 


## ✨ Features

List the key features of your project using bullet points.

* **Exploratory Data Analysis (EDA):** Charts and statistics offering deep insights into the car dataset.
* **Machine Learning Models:** Utilizes ML algorithms to train and build accurate car price prediction models.
* **Prediction API:** A RESTful API using FastAPI for integrating car price predictions into other applications.
* **Web Interface:** A simple web to allow users to input car information and receive instant price predictions.
* **Containerization (Docker):** Ensures easy deployment across different platforms.

## 🚀 Project Structure

Describe your project's directory structure to help readers quickly locate important files.

.
├── .venv/                      # Python virtual environment
├── app/                        # Directory containing the main API application
│   ├── pycache/
│   └── api.py                  # FastAPI API endpoints
├── notebooks/                  # Jupyter Notebooks for EDA and model experimentation
│   ├── eda.ipynb                                   # Main EDA file
│   ├── main-draft.ipynb                            # A draft contains some notes about the project and pipeline
│   └── preprocessing-and-model-experiment.ipynb    # A main notebook contains the project preprocessing pipeline and models.
├── src/                        # Source code for the chosen model
│   ├── pycache/
│   ├── .gradio/                # Contains Gradio-related configurations or files
│   ├── best-model-pipeline.ipynb # Notebook with the optimized model pipeline
│   ├── data_preprocessor.py    # Data preprocessing script
│   ├── model.pkl               # Trained machine learning model
│   ├── preprocessor.pkl        # Saved preprocessor object
│   └── prototype.ipynb         # Gradio prototype for model testing 
├── .gitignore                  
├── car-dataset.csv             # The car dataset used for training
├── Dockerfile                  # Docker configuration file 
└── requirements.txt            # List of required Python libraries

## 🛠️ Tech Stack

List the main languages, frameworks, and libraries.

* **Language:** Python
* **API Framework:** FastAPI (or Flask, if used)
* **ML Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, category_encoders,...
* **Deployment:** Docker, Uvicorn
* **Web Interface:** HTML, CSS, JS
