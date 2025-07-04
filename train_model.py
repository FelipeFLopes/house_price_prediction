import json
import pathlib
import pickle

import mlflow
from mlflow.models import infer_signature
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import root_mean_squared_error


from create_model import SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION, OUTPUT_DIR, load_data


sales_demographic_data, target = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)


x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=42)


params = {}

model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                neighbors.KNeighborsRegressor(**params)).fit(
                                    x_train, y_train)




y_pred = model.predict(x_test)

mse_test = root_mean_squared_error(y_test, y_pred)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Client Pipeline")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("RMSE", mse_test)

    # Infer the model signature
    signature = infer_signature(x_train, model.predict(x_train))

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="knn",
        signature=signature,
        input_example=x_train,
        registered_model_name="knn_client",
    )


output_dir = pathlib.Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

# Output model artifacts: pickled model and JSON list of features
pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
json.dump(list(x_train.columns),
            open(output_dir / "model_features.json", 'w'))