import requests
import json

import pandas as pd


PREDICTION_API_ENDPOINT = "http://127.0.0.1:8000/predict"
PREDICTION_API_SUBSET_ENDPOINT = "http://127.0.0.1:8000/predict_from_subset"

BASE_MODEL_NAME = "knn_with_imputer"
SUBSET_MODEL_NAME = "knn_min_columns"


def test_predict_endpoint(url, model_name, model_version):

    COLUMNS = ['sqft_lot', 'floors', 'sqft_living', 'sqft_basement', 'bathrooms', 'bedrooms', 'sqft_above']

    path_unseen_data = "data/future_unseen_examples.csv"

    dataset = pd.read_csv(path_unseen_data)

    data = dataset[COLUMNS].to_dict(orient="records")

    payload = {"model_name":model_name, "model_version": model_version, "dataframe_records": data}

    res = requests.post(url=url, data=json.dumps(payload))

    return res.json()


if __name__ == "__main__":

    res = test_predict_endpoint(url=PREDICTION_API_SUBSET_ENDPOINT, model_name=SUBSET_MODEL_NAME, model_version="latest")

    print(res)