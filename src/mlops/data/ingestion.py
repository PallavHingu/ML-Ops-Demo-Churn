from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
from mlops.core.base_component import BaseComponent

class DataIngestor(BaseComponent):
    """Pulls data from BigQuery"""

    def run(self) -> pd.DataFrame:
        client = bigquery.Client(project="mlops-demo-churn")
        query = f"SELECT * FROM `mlops-demo-churn.churn_demo.raw`"
        self.log.info("Executing query: %s", query)
        df = client.query(query).to_dataframe()

        self.log.info("Ingested %d rows", len(df))
        return df