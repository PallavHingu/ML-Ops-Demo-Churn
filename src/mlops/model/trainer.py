from sklearn.ensemble import GradientBoostingClassifier
import joblib
from mlops.core.base_component import BaseComponent
import numpy as np, tempfile

class ModelTrainer(BaseComponent):
    """Trains the model using input data. Trains a Gradient Boost Classifier model."""
    def run(self, X: np.ndarray, y: np.ndarray):
        self.log.info("TrainingGradientBoosting Classifier …")
        print("Training Gradient Boosting Classifier …")
        print("..mm.")
        clf = GradientBoostingClassifier(random_state=42)
        print(".n.")
        print(str(type(X)))
        print(str(type(y)))
        clf.fit(X, y)
        print("Trained GradientBoostingClassifier…")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as fp:
            joblib.dump(clf, fp.name)
        return clf, fp.name