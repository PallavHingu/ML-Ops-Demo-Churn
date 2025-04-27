import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlops.core.base_component import BaseComponent
import joblib, tempfile

class DataPreprocessor(BaseComponent):
    """Cleans & encodes features; outputs X, y and fitted pipeline object."""

    def run(self, df: pd.DataFrame):
        self.log.info("Starting preprocessing …")
        df = df.copy()
        df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

        pre = ColumnTransformer(
            transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
            ],
            sparse_threshold=0
        )
        pipe = Pipeline([('pre', pre)])
        X_processed = pipe.fit_transform(X)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as fp:
            joblib.dump(pipe, fp.name)
        self.log.info("Preprocessing complete; shape=%s", X_processed.shape)
        return X_processed, y.values, fp.name