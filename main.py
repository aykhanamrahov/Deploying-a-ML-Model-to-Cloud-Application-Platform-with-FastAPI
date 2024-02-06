import pandas as pd
import numpy as np
import joblib
import os

from fastapi import FastAPI
from model import InputData
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data, categorical_features

MODEL_FOLDER_PATH = os.getcwd() + "/starter/model/"

# Create Fast API app
app = FastAPI()
# Load the ML model
model = joblib.load(MODEL_FOLDER_PATH + "model.pkl")
encoder = joblib.load(MODEL_FOLDER_PATH + "encoder.pkl")
lb = joblib.load(MODEL_FOLDER_PATH + "lb.pkl")


# Create API
@app.get('/')
async def check_heath():
    return {"I am still alive ヾ(⌐■_■)ノ♪"}


@app.post('/predict')
async def predict(request_data: InputData):
    print("request_data:", request_data)
    # Get the input data from the request data
    data = {
        "age": request_data.age,
        "workclass": request_data.workclass,
        "fnlwgt": request_data.fnlgt,
        "education": request_data.education,
        "education-num": request_data.education_num,
        "marital-status": request_data.marital_status,
        "occupation": request_data.occupation,
        "relationship": request_data.relationship,
        "race": request_data.race,
        "sex": request_data.sex,
        "capital-gain": request_data.capital_gain,
        "capital-loss": request_data.capital_loss,
        "hours-per-week": request_data.hours_per_week,
        "native-country": request_data.native_country
    }
    data_input = pd.DataFrame([data])
    # Process the data input
    data_pred, _, _, _ = process_data(data_input, categorical_features=categorical_features, training=False,
                                      encoder=encoder, lb=lb)
    # Inference model
    result_pred = inference(model=model, X=data_pred)
    return lb.inverse_transform(result_pred)[0]
