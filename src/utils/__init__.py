"""Utility modules for ML pipeline DAGs.

This module exposes the lightweight, commonly-used symbols from the
`utils` package. Heavy dependencies (TensorFlow, MLflow, etc.) are
lazy-loaded when accessed to avoid import-time side effects.
"""

from .config_load import Config
from .data_transform import (
	PipelineImputer,
	PipelineOneHotEncoder,
	PipelineFeatureDropper,
	PipelineFeatureStandardScaler,
	PipelineDataSplit,
	DatasplitToNumpyArray,
	PipelineDateSpliter,
	PipelineIndexSetter,
	PipelineSequencer,
)
from .data_pipeline import build_pipeline, save_pipeline, load_pipeline
from .mlflow_log import MLFlowLogger

# Public names that are safe to import from the package.
__all__ = [
	"Config",
	"PipelineImputer",
	"PipelineOneHotEncoder",
	"PipelineFeatureDropper",
	"PipelineFeatureStandardScaler",
	"PipelineDataSplit",
	"DatasplitToNumpyArray",
	"PipelineDateSpliter",
	"PipelineIndexSetter",
	"PipelineSequencer",
	"build_pipeline",
	"save_pipeline",
	"load_pipeline",
	"MLFlowLogger",
	# heavy modules exposed lazily
	"model_template",
]


def __getattr__(name: str):
	"""Lazy-load heavy submodules on attribute access.

	This avoids importing TensorFlow/MLflow during simple package
	imports like ``from utils import Config``.
	"""
	if name in ("model_template",):
		import importlib

		module = importlib.import_module(f".{name}", __name__)
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	return sorted(list(globals().keys()) + __all__)
