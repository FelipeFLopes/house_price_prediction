import requests
import json

import pandas as pd


PREDICTION_API_ENDPOINT = "http://127.0.0.1:8000/predict"

COLUMNS = ['sqft_lot', 'floors', 'sqft_living', 'sqft_basement', 'bathrooms', 'bedrooms', 'sqft_above']


path_unseen_data = "data/future_unseen_examples.csv"

dataset = pd.read_csv(path_unseen_data)

data = dataset[COLUMNS].to_dict(orient="records")

payload = {"model_name":"knn_with_imputer", "model_version": "latest", "dataframe_records": data}

res = requests.post(url=PREDICTION_API_ENDPOINT, data=json.dumps(payload))

print(res.json())