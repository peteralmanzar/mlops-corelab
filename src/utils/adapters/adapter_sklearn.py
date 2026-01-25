from typing import Any, Dict
import pickle
import os
import time


class SklearnAdapter:
    def __init__(self, estimator=None, params: Dict[str, Any] = None):
        self.params = params or {}
        self.estimator = estimator
        self._is_trained = False

    def _build_estimator(self):
        # lazy import here in case sklearn not installed at import time
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**self.params)

    def fit(self, X, y, **fit_kwargs):
        if self.estimator is None:
            self.estimator = self._build_estimator()
        start = time.time()
        self.estimator.fit(X, y, **fit_kwargs)
        elapsed = time.time() - start
        self._is_trained = True
        return {"training_time_sec": elapsed}

    def predict(self, X):
        if not self.estimator:
            raise RuntimeError("Estimator not initialized")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError("predict_proba not available for this estimator")
        return self.estimator.predict_proba(X)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.estimator, f)
        return path

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            est = pickle.load(f)
        return cls(estimator=est, params=getattr(est, "get_params", lambda: {})())

    def get_params(self):
        if self.estimator and hasattr(self.estimator, "get_params"):
            return self.estimator.get_params()
        return self.params or {}


__all__ = ["SklearnAdapter"]
