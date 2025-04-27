from sklearn.ensemble import GradientBoostingClassifier
import joblib
from mlops.core.base_component import BaseComponent
import numpy as np, tempfile

class ModelTrainer(BaseComponent):
    """Trains the model using input data. Trains a Gradient Boost Classifier model."""
    def run(self, X: np.ndarray, y: np.ndarray):
        self.log.info("Training GradientBoostingClassifier…")
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X, y)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as fp:
            joblib.dump(clf, fp.name)
        return clf, fp.name