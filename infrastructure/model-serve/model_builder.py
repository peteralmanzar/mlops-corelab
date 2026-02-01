"""
Model builder module for unpickling sklearn pipelines.

Contains KerasModelWrapper which is referenced in pickled combined pipelines.
This is a minimal version for serving - the full version is in src/utils/.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class KerasModelWrapper(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for a trained Keras model.
    Used so the Keras model can be the final step in an sklearn Pipeline.
    """
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure numpy array
        arr = X.values if hasattr(X, 'values') else np.asarray(X)
        preds = self.model.predict(arr)
        return preds

    def predict(self, X):
        """Predict method for use in sklearn pipeline."""
        arr = X.values if hasattr(X, 'values') else np.asarray(X)
        preds = self.model.predict(arr)
        # Flatten if single output column
        if preds.ndim == 2 and preds.shape[1] == 1:
            return preds.flatten()
        return preds


# For backward compatibility when unpickling
__all__ = ["KerasModelWrapper"]
