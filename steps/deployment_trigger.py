
from zenml import step
from zenml.steps import BaseParameters

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.0


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy