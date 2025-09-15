import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import numpy as np
import joblib
from pathlib import Path

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories, save_json

# Set MLflow environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/purnachandrareddy2003/DataScienceProject.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "purnachandrareddy2003"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "361f48b765578ba2d34f7a70d481015de6fdce18"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally as JSON
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # ✅ Save model manually as artifact
            model_dir = Path("artifacts/model_evaluation/model")
            create_directories([model_dir])  # Ensure directory exists

            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)

            # ✅ Correct indentation here
            mlflow.log_artifact(str(model_path), artifact_path="model")
