import logging
import pandas as pd
import mlflow
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from model.model_dev import HyperparameterTuner, LightGBMModel, LinearRegressionModel, RandomForestModel, XGBoostModel
from steps.config import ModelNameConfig

# Set up the experiment tracker from the active ZenML stack
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
    model_stage: str = "Staging"
) -> RegressorMixin:
    """
    Train a regression model based on the specified configuration.

    Args:
        x_train (pd.DataFrame): Training features.
        x_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        config (ModelNameConfig): Configuration object with model details.

    Returns:
        RegressorMixin: The trained regression model.
    """
    try:
        model = None

        # Select the model and enable autologging
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        # Initialize the hyperparameter tuner
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        # Train the model with or without hyperparameter tuning
        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        """
        # Log the model and metrics with MLflow
        with mlflow.start_run(nested=True) as run:
            # Log parameters and metrics
            mlflow.log_param("model_name", config.model_name)
            mlflow.log_param("fine_tuning", config.fine_tuning)
            # Example metric, replace with your actual metric calculation
            mse = ((y_test - trained_model.predict(x_test)) ** 2).mean()  # Replace with actual metric calculation
            mlflow.log_metric("mse", mse)

            # Log the trained model
            mlflow.sklearn.log_model(trained_model, "model")
            # Register the model in the model registry
            registered_model=mlflow.register_model(f"runs:/{run.info.run_id}/model", "model", tags={"stage": model_stage})
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=registered_model.name,
                version=registered_model.version,
                stage=model_stage
            )
        """
        return trained_model
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise e