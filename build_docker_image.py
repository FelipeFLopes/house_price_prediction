import mlflow


mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")

mlflow.models.build_docker(
    model_uri=f"models:/knn_with_imputer/latest",
    name="knn_with_imputer"
)
