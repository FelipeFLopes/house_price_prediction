
import requests
import json
from typing import Any, Optional, Union
from pydantic import BaseModel

from fastapi import FastAPI
import mlflow



mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")
app = FastAPI()




class PredictApiData(BaseModel):
    input_image: Any
    model_name: str




@app.post("/predict")
async def predict_api():

    model_name = "knn_with_imputer"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"


    model_info = mlflow.models.get_model_info(model_uri)

    signature = model_info.signature


    model_inputs_schema = signature.inputs.to_dict()


    model_input_column = []
    for column in model_inputs_schema:

        model_input_column.append(column["name"])


    payload = {"columns":[], "data": []}
    for column in model_input_column:

        payload["columns"].append(column)

        payload["data"].append(0)


    payload["data"] = [payload["data"]]

    url = "http://localhost:5003/invocations"

    
    payload_formated = {}

    payload_formated["dataframe_split"] = payload

    print(payload_formated)

    headers = {
        "Content-Type": "application/json",
    }

    payload_json = json.dumps(payload_formated)

    print(payload_json)

    params = {
       "dataframe_split" :  payload
    }

    params = json.dumps(params)

    res = requests.post(url=url, data=params, headers=headers)

    print(res.json())
