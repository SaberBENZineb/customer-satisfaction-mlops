from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.steps import BaseParameters

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True,
    )
    predictor(service=model_deployment_service, data=batch_data)