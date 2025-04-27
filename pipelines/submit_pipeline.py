import os
from google.cloud import aiplatform

def submit_pipeline():
    project = os.environ.get("GCP_PROJECT")
    region = os.environ.get("REGION", "europe-west2")
    staging_bucket = f"gs://{project}-pipeline-root"
    pipeline_file = "churn_pipeline.yaml"
    service_account = f"vertex-pipelines-sa@{project}.iam.gserviceaccount.com"

    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    job = aiplatform.PipelineJob(
        display_name="churn-pipeline-ci",
        template_path=pipeline_file,
        pipeline_root=staging_bucket,
        service_account=service_account,
        enable_caching=True,
    )
    job.run(sync=True)

if __name__ == "__main__":
    submit_pipeline()