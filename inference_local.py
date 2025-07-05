import pandas as pd

import mlflow.sklearn

path_unseen_data = "data/future_unseen_examples.csv"

dataset = pd.read_csv(path_unseen_data)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

model_name = "knn_client"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"


model_info = mlflow.models.get_model_info(model_uri)
signature = model_info.signature


model_inputs_schema = signature.inputs.to_dict()


model_input_column = []
for column in model_inputs_schema:

    model_input_column.append(column["name"])


print(model_input_column)

columns_dataset = dataset.columns.values


columns_missing_input_dataset = set(model_input_column) - set(columns_dataset)
columns_excess_input_dataset = set(columns_dataset) - set(model_input_column)


dataset_clean_exccess_columns = dataset.drop(columns=columns_excess_input_dataset)

dataset_filled = dataset_clean_exccess_columns.copy()
for column in columns_missing_input_dataset:

    dataset_filled[column] = 0


print(dataset_filled.columns)

dataset_filled_reordered = dataset_filled[model_input_column]


model = mlflow.sklearn.load_model(model_uri)
pred = model.predict(dataset_filled_reordered)


print(pred)
