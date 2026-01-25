from typing import Any, Dict, Optional
import os
import time


class KerasAdapter:
    def __init__(self, model=None, template: Optional[str] = None, template_args: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
        self._model = model
        self.template = template
        self.template_args = template_args or {}
        self.params = params or {}

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        # Lazy imports
        try:
            from tensorflow.keras.models import Model
        except Exception as e:
            raise RuntimeError("TensorFlow/Keras is required for KerasAdapter but is not available: %s" % e)

        # If a template name is provided, call the corresponding function in model_template
        if self.template:
            from utils import model_template as mt

            fn = getattr(mt, self.template, None)
            if fn is None:
                raise ValueError(f"Keras template not found: {self.template}")
            self._model = fn(**self.template_args)
            return self._model

        raise RuntimeError("No Keras model or template provided to KerasAdapter")

    def fit(self, X, y, **fit_kwargs):
        model = self._ensure_model()
        start = time.time()
        history = model.fit(X, y, **fit_kwargs)
        elapsed = time.time() - start
        return {"training_time_sec": elapsed, "history": getattr(history, "history", {})}

    def predict(self, X):
        model = self._ensure_model()
        preds = model.predict(X)
        return preds

    def predict_proba(self, X):
        # for Keras, predict returns probabilities for many models
        return self.predict(X)

    def save(self, path: str) -> str:
        model = self._ensure_model()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Use Keras's save
        model.save(path)
        return path

    @classmethod
    def load(cls, path: str):
        try:
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError("TensorFlow/Keras is required to load Keras models: %s" % e)
        model = load_model(path)
        return cls(model=model)

    def get_params(self):
        return self.params or {}


__all__ = ["KerasAdapter"]
