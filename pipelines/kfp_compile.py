from kfp import compiler
from churn_pipeline import churn_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(churn_pipeline, "churn_pipeline.yaml")