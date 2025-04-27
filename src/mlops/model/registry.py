from google.cloud import aiplatform
from mlops.core.base_component import BaseComponent

class ModelRegistrar(BaseComponent):
    def run(self, model_artifact: str):
        aiplatform.init(project="mlops-demo-churn", location="europe-west2", staging_bucket="gs://mlops-demo-churn-pipeline-root")
        model = aiplatform.Model.upload(
            display_name=churn-model,
            artifact_uri=model_artifact,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest")
        self.log.info("Model registered: %s", model.resource_name)
        return model.resource_name