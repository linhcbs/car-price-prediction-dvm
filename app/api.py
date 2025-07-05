from fastapi import FastAPI
import sys
sys.path.append('./src') 
from data_preprocessor import DataPreprocessor
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel



# KHOI TAO API
app = FastAPI(title='Car Prediction API', description = 'API for predicting car prices based on various features by using a trained model :)', version = '1.0.0')

# LOAD MODEL

with open('./src/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('./src/model.pkl', 'rb') as f:
    model = pickle.load(f)


# DINH NGHIA CAU TRUC DU LIEU DAU VAO
class CarFeatures (BaseModel):
    age: float
    runned_miles: float
    engine_size: float
    engine_power: float
    width: float
    length: float
    average_mpg: float
    top_speed: float
    seat_num: int
    maker: str
    genmodel: str
    bodytype: str
    gearbox: str
    fuel_type: str


# TAO ENDPOINT DE KIEM TRA HOAT DONG
@app.get('/')
def read_root() :
    return {'message': 'Welcome to Car Price Prediction API! Go to /docs to see API documentation'}


# TAO ENDPOINT DE LAY THONG TIN DU LIEU PHAN LOAI
@app.get('/categorical-choices')
def send_categorical_choices ():
    return {
        'choices': preprocessor.get_categorical_choices()
    }



# TAO ENDPOINT DE PREDICT
@app.post('/predict')
def predict(features: CarFeatures) :
    if model is None: 
        return {'error': 'Model not loaded. Cannot make predictions'}
    

    # chuyen du lieu tu pydantic thanh df
    inp = {
        'age': [features.age],
        'runned-miles': [features.runned_miles],
        'engine-size': [features.engine_size],
        'engine-power': [features.engine_power],
        'width': [features.width],
        'length': [features.length],
        'average-mpg': [features.average_mpg],
        'top-speed': [features.top_speed],
        'seat-num': [features.seat_num],
        'maker': [features.maker],
        'genmodel': [features.genmodel],
        'bodytype': [features.bodytype],
        'gearbox': [features.gearbox],
        'fuel-type': [features.fuel_type]
    }
    X = pd.DataFrame(inp)

    # # chuan bi du lieu
    X = preprocessor.perform_light_cleaning(X)
    X = preprocessor.impute(X)
    X = preprocessor.encode(X)  
    X = X[preprocessor.get_encoded_cols()]  
    X = preprocessor.scale(X)

    # # du doan
    y_pred = model.predict(X)
    pred = y_pred[0].astype('float64')

    # tra ket qua
    return {
        "prediction": pred,
    }


# TAO ENDPOINT DE LAY THONG TIN DU LIEU PHAN LOAI
@app.get('/categorical-choices')
def send_categorical_choices ():
    return {
        'choices': preprocessor.get_categorical_choices()
    }


# ! uvicorn app.api:app --reload
# swagger: /docs