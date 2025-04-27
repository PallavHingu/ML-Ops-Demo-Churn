"""KFP v2 pipeline: wrapper components call reusable classes.
Each component is a top‑level function so that the KFP SDK can
introspect inputs/outputs correctly.  No pandas.DataFrame objects are
returned directly – instead we pass artefact files (CSV / Joblib),
which keeps the pipeline serialisable and avoids the
`DataFrame.to_dict()` error you hit.
"""

from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Artifact

# ---------- wrappers --------------------------------------------------

@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=[
    #     "mlops",              # our repo (already in image but needed when local)
    #     "pandas",
    #     "google-cloud-bigquery",
    #     "google-cloud-storage",
    #     "pyarrow"
    # ],
)
def ingest_component(output_data: Output[Dataset]):
    """Reads BigQuery → writes CSV artefact."""
    import pandas as pd
    from mlops.data.ingestion import DataIngestor

    df = DataIngestor().run()
    df.to_csv(output_data.path, index=False)   # KFP uploads file


@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=[
    #     "mlops",
    #     "pandas",
    #     "scikit-learn",
    #     "joblib"
    # ],
)
def preprocess_component(
    raw_data: Input[Dataset],
    X_out: Output[Artifact],
    y_out: Output[Artifact],
    prep_pipe: Output[Artifact],
):
    import pandas as pd, numpy as np, joblib, tempfile
    from mlops.data.preprocessing import DataPreprocessor

    df = pd.read_csv(raw_data.path)
    pre = DataPreprocessor()
    X, y, pipe_path = pre.run(df)

    # Save numpy arrays for downstream steps
    np.save(X_out.path, X)
    np.save(y_out.path, y)
    joblib.copy(pipe_path, prep_pipe.path)


@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=["mlops", "scikit-learn", "joblib", "numpy"],
)
def train_component(
    X_in: Input[Artifact],
    y_in: Input[Artifact],
    model_artifact: Output[Artifact],
):
    import numpy as np, joblib, os
    from mlops.model.trainer import ModelTrainer

    X = np.load(X_in.path, allow_pickle=True)
    y = np.load(y_in.path, allow_pickle=True)
    trainer = ModelTrainer()
    _clf, model_path = trainer.run(X, y)
    joblib.copy(model_path, model_artifact.path)


@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=["mlops", "scikit-learn", "joblib", "numpy"],
)
def evaluate_component(
    model_artifact: Input[Artifact],
    X_in: Input[Artifact],
    y_in: Input[Artifact],
    accuracy: Output[Artifact],
):
    import joblib, numpy as np, json
    from mlops.model.evaluator import ModelEvaluator

    clf = joblib.load(model_artifact.path)
    X = np.load(X_in.path, allow_pickle=True)
    y = np.load(y_in.path, allow_pickle=True)
    evaluator = ModelEvaluator()
    metrics, metrics_path = evaluator.run(clf, X, y)
    # json.dump(metrics, open(metrics_out.path, "w"))
    joblib.copy(metrics, accuracy)



@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=["mlops", "google-cloud-aiplatform"],
)
def register_component(model_artifact: Input[Artifact]) -> str:
    from mlops.model.registry import ModelRegistrar
    registrar = ModelRegistrar()
    return registrar.run(model_artifact.path)


@component(
    base_image="europe-west2-docker.pkg.dev/mlops-demo-churn/mlops-custom-images/churn-mlops-base:latest",
    # packages_to_install=["mlops", "google-cloud-aiplatform"],
)
def deploy_component(model_name: str) -> str:
    from mlops.deploy.deployer import ModelDeployer
    return ModelDeployer().run(model_name)

# ---------- pipeline --------------------------------------------------

@dsl.pipeline(name="churn_pipeline", description="End‑to‑end churn prediction (Vertex)")
def churn_pipeline():
    ingest = ingest_component()
    preprocess = preprocess_component(raw_data=ingest.outputs["output_data"])

    train = train_component(
        X_in=preprocess.outputs["X_out"],
        y_in=preprocess.outputs["y_out"],
    )

    evaluate = evaluate_component(
        model_artifact=train.outputs["model_artifact"],
        X_in=preprocess.outputs["X_out"],
        y_in=preprocess.outputs["y_out"],
    )

    # threshold = 0.85

    # with dsl.If(evaluate.outputs["accuracy"] > 0.85):
    #     register = register_component(model_artifact=train.outputs["model_artifact"])
    #     deploy = deploy_component(model_name=register.output)