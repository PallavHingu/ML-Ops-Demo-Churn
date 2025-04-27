from sklearn.ensemble import GradientBoostingClassifier
import joblib
from mlops.core.base_component import BaseComponent
import numpy as np, tempfile

class ModelTrainer(BaseComponent):
    """Trains the model using input data. Trains a Gradient Boost Classifier model."""
    def run(self, X: np.ndarray, y: np.ndarray):
        self.log.info("Training GradientBoostingClassifier…")
        self.log.info("...")
        clf = GradientBoostingClassifier(random_state=42)
        self.log.info("..")
        self.log.info(str(type(X)))
        self.log.info(str(type(y)))
        clf.fit(X, y)
        self.log.info("Trained GradientBoostingClassifier…")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as fp:
            joblib.dump(clf, fp.name)
        return clf, fp.name