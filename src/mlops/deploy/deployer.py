from google.cloud import aiplatform
from mlops.core.base_component import BaseComponent

class ModelDeployer(BaseComponent):
    def run(self, model_name: str):
        aiplatform.init(project="mlops-demo-churn", location="europe-west2")
        endpoint = aiplatform.Endpoint.create(display_name="churn-endpoint")
        endpoint.deploy(model=model_name, traffic_percentage=100, machine_type="n1-standard-2")
        self.log.info("Model deployed to endpoint: %s", endpoint.resource_name)
        return endpoint.resource_name