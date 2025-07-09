
import requests
import json
from typing import Any
from pydantic import BaseModel

from fastapi import FastAPI
import mlflow
import numpy as np
import pandas as pd


MODEL_PRED_ENDPOINT_URL = "http://localhost:5003/invocations"
MODEL_PRED_SUBSET_ENDPOINT_URL = "http://localhost:5004/invocations"
TRACKING_SERVER_URI = "sqlite:///mlruns.db"

mlflow.set_tracking_uri(uri=TRACKING_SERVER_URI)
app = FastAPI()


class PredictApiData(BaseModel):
    model_name: Any
    model_version: str
    dataframe_records: list[dict]





@app.post("/predict")
def predict_api(data: PredictApiData):

    model_name = data.model_name
    model_version = data.model_version
    model_uri = f"models:/{model_name}/{model_version}"


    model_info = mlflow.models.get_model_info(model_uri)
    signature = model_info.signature
    model_inputs_schema = signature.inputs.to_dict()

    model_input_column = []
    for column in model_inputs_schema:

        model_input_column.append(column["name"])



    input = pd.DataFrame.from_records(data.dataframe_records)
    columns_dataset = input.columns.values

    columns_missing_input_dataset = set(model_input_column) - set(columns_dataset)


    for column_missing in columns_missing_input_dataset:
        input[column_missing] = np.nan

    dataframe_with_missing_columns = input.to_dict("split")


    payload = {"columns": dataframe_with_missing_columns["columns"], "data": dataframe_with_missing_columns["data"]}


    headers = {
        "Content-Type": "application/json",
    }

    params = {
       "dataframe_split" :  payload
    }
    params = json.dumps(params)


    res = requests.post(url=MODEL_PRED_ENDPOINT_URL, data=params, headers=headers)


    return res.json()


@app.post("/predict_from_subset")
def predict_subset_columns(data: PredictApiData):

    model_name = data.model_name
    model_version = data.model_version
    model_uri = f"models:/{model_name}/{model_version}"


    model_info = mlflow.models.get_model_info(model_uri)
    signature = model_info.signature
    model_inputs_schema = signature.inputs.to_dict()

    model_input_column = []
    for column in model_inputs_schema:

        model_input_column.append(column["name"])



    input = pd.DataFrame.from_records(data.dataframe_records)
    columns_dataset = input.columns.values

    columns_missing_input_dataset = set(model_input_column) - set(columns_dataset)


    for column_missing in columns_missing_input_dataset:
        input[column_missing] = np.nan

    dataframe_with_missing_columns = input.to_dict("split")


    payload = {"columns": dataframe_with_missing_columns["columns"], "data": dataframe_with_missing_columns["data"]}


    headers = {
        "Content-Type": "application/json",
    }

    params = {
       "dataframe_split" :  payload
    }
    params = json.dumps(params)


    res = requests.post(url=MODEL_PRED_SUBSET_ENDPOINT_URL, data=params, headers=headers)


    return res.json()