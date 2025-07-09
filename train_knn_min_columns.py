import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

from create_model import SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION, load_data
from utils import prediction_correctness


sales_demographic_data, target = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)


x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

x_minimal = x[['sqft_above', 'floors', 'sqft_basement', 'bathrooms', 'bedrooms', 'sqft_lot', 'sqft_living']]

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_minimal, y, random_state=42)


params = {}

model = pipeline.make_pipeline(SimpleImputer(),
                               preprocessing.RobustScaler(),
                                neighbors.KNeighborsRegressor(**params)).fit(
                                    x_train, y_train)



y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

mrse_train = root_mean_squared_error(y_train, y_pred_train)
mrse_test = root_mean_squared_error(y_test, y_pred_test)

predictions_class_train = pd.Series(prediction_correctness(y_train, y_pred_train))
predictions_class_test = pd.Series(prediction_correctness(y_test, y_pred_test))


normalized_predction_class_train = predictions_class_train.value_counts(normalize=True).to_dict()
normalized_predction_class_test = predictions_class_test.value_counts(normalize=True).to_dict()




mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")
mlflow.set_experiment("Client base model")


with mlflow.start_run():

    mlflow.log_params(params)

    mlflow.log_metric("RMSE train", mrse_train)
    mlflow.log_metric("Correct train", normalized_predction_class_train["Correct"])
    mlflow.log_metric("Overstimate train", normalized_predction_class_train["Overestimate"])
    mlflow.log_metric("Understimate train", normalized_predction_class_train["Underestimate"])

    mlflow.log_metric("RMSE test", mrse_test)
    mlflow.log_metric("Correct test", normalized_predction_class_test["Correct"])
    mlflow.log_metric("Overstimate test", normalized_predction_class_test["Overestimate"])
    mlflow.log_metric("Understimate test", normalized_predction_class_test["Underestimate"])


    signature = infer_signature(x_train, model.predict(x_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="knn",
        signature=signature,
        input_example=x_train,
        registered_model_name="knn_min_columns",
    )
