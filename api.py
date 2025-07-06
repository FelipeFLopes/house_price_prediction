
import requests
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
async def predict_api(data: PredictApiData):

    model_name = "knn_with_imputer"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"


    model_info = mlflow.models.get_model_info(model_uri)

    signature = model_info.signature


    model_inputs_schema = signature.inputs.to_dict()


    model_input_column = []
    for column in model_inputs_schema:

        model_input_column.append(column["name"])


    print(model_input_column)


    payload = {}
    for column in model_input_column:
        payload[column] = 0

    requests.post()

    return {"result": res}