"""Utility modules for ML pipeline DAGs.

This module exposes the lightweight, commonly-used symbols from the
`utils` package. Heavy dependencies (TensorFlow, MLflow, etc.) are
lazy-loaded when accessed to avoid import-time side effects.
"""

from .data_transform import (
	PipelineOneHotEncoder,
	PipelineFeatureDropper,
	PipelineFeatureStandardScaler,
	PipelineDataSplit,
	DatasplitToNumpyArray,
	PipelineDateSpliter,
	PipelineIndexSetter,
	PipelineSequencer,
)
from .data_validate import DataValidator
from .mlflow_log import MLflowLogger
from .pipeline_preprocess import build_pipeline, save_pipeline, load_pipeline

# Public names that are safe to import from the package.
__all__ = [
	"PipelineOneHotEncoder",
	"PipelineFeatureDropper",
	"PipelineFeatureStandardScaler",
	"PipelineDataSplit",
	"DatasplitToNumpyArray",
	"PipelineDateSpliter",
	"PipelineIndexSetter",
	"PipelineSequencer",
	"DataValidator",
	"MLflowLogger",
	"build_pipeline",
	"save_pipeline",
	"load_pipeline",
	# heavy modules exposed lazily
	"model_template",
	"model_register",
]


def __getattr__(name: str):
	"""Lazy-load heavy submodules on attribute access.

	This avoids importing TensorFlow/MLflow during simple package
	imports like ``from utils import DataValidator``.
	"""
	if name in ("model_template", "model_register"):
		import importlib

		module = importlib.import_module(f".{name}", __name__)
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	return sorted(list(globals().keys()) + __all__)
