from sklearn.metrics import accuracy_score, roc_auc_score
from mlops.core.base_component import BaseComponent
import numpy as np, json, tempfile

class ModelEvaluator(BaseComponent):
    def run(self, clf, X_valid: np.ndarray, y_valid: np.ndarray):
        preds = clf.predict(X_valid)
        proba = clf.predict_proba(X_valid)[:,1]
        acc = accuracy_score(y_valid, preds)
        auc = roc_auc_score(y_valid, proba)
        metrics = {'accuracy': acc, 'roc_auc': auc}
        self.log.info("Eval: %s", metrics)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as fp:
            json.dump(metrics, fp)
        return metrics, fp.name