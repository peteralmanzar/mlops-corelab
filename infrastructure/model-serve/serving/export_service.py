"""
Model Export Service

Packages a loaded champion model into a self-contained
FastAPI Docker project as a downloadable zip file.
"""

import json
import shutil
import logging
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import joblib

from .model_manager import ModelManager, LoadedModel

logger = logging.getLogger(__name__)

# Template directory lives alongside the serving package
TEMPLATE_DIR = Path(__file__).parent.parent / "export_template"
# Custom modules needed for sklearn unpickling
CUSTOM_MODULES_DIR = Path(__file__).parent.parent


class ModelExportService:
    """Packages a loaded model into a standalone FastAPI Docker project."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def export_model(self, model_name: str) -> Path:
        """
        Export a loaded champion model as a standalone Docker package.

        Args:
            model_name: Name of the model to export (must be currently loaded)

        Returns:
            Path to the generated zip file

        Raises:
            ValueError: If model is not found
        """
        loaded_model = self.model_manager.get_model(model_name)
        if loaded_model is None:
            available = [m["name"] for m in self.model_manager.list_models()]
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available}"
            )

        logger.info(f"Starting export for model: {model_name} v{loaded_model.version}")

        # Create temp directory for the export
        tmp_base = Path(tempfile.mkdtemp(prefix="model_export_"))
        export_dir = tmp_base / f"{model_name}_standalone"
        export_dir.mkdir(parents=True)

        try:
            # 1. Copy template files
            self._copy_template(export_dir)

            # 2. Copy custom transformer modules for unpickling
            self._copy_custom_modules(export_dir)

            # 3. Serialize model artifact to disk
            self._save_model_artifact(loaded_model, export_dir)

            # 4. Write model metadata
            self._write_model_metadata(loaded_model, export_dir)

            # 5. Generate config.json with model section pre-filled
            self._write_config(loaded_model, export_dir)

            # 6. Create zip
            zip_path = self._create_zip(export_dir, model_name, tmp_base)

            logger.info(f"Export complete: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return zip_path

        except Exception:
            # Clean up on failure
            shutil.rmtree(tmp_base, ignore_errors=True)
            raise

    def _copy_template(self, export_dir: Path) -> None:
        """Copy template app files to export directory."""
        logger.info("Copying template files...")

        # Copy app/ directory
        src_app = TEMPLATE_DIR / "app"
        dst_app = export_dir / "app"
        if src_app.exists():
            shutil.copytree(src_app, dst_app)

        # Copy Dockerfile and requirements.txt
        for filename in ("Dockerfile", "requirements.txt"):
            src = TEMPLATE_DIR / filename
            if src.exists():
                shutil.copy2(src, export_dir / filename)

    def _copy_custom_modules(self, export_dir: Path) -> None:
        """Copy data_transform.py and model_builder.py for sklearn unpickling."""
        logger.info("Copying custom transformer modules...")

        lib_dir = export_dir / "lib"
        lib_dir.mkdir(exist_ok=True)

        for module_name in ("data_transform.py", "model_builder.py"):
            src = CUSTOM_MODULES_DIR / module_name
            if src.exists():
                shutil.copy2(src, lib_dir / module_name)
                logger.info(f"  Copied {module_name}")
            else:
                logger.warning(f"  Module not found: {src}")

    def _save_model_artifact(self, loaded_model: LoadedModel, export_dir: Path) -> None:
        """Serialize the loaded model to disk using joblib."""
        logger.info(f"Serializing model ({loaded_model.model_type})...")

        model_dir = export_dir / "model"
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "model.pkl"
        joblib.dump(loaded_model.model, model_path, compress=3)
        size_mb = model_path.stat().st_size / 1024 / 1024
        logger.info(f"  Model saved: {size_mb:.1f} MB")

    def _write_model_metadata(self, loaded_model: LoadedModel, export_dir: Path) -> None:
        """Write model metadata JSON."""
        logger.info("Writing model metadata...")

        model_dir = export_dir / "model"
        model_dir.mkdir(exist_ok=True)

        metadata = {
            "name": loaded_model.name,
            "version": loaded_model.version,
            "run_id": loaded_model.run_id,
            "model_type": loaded_model.model_type,
            "task_type": loaded_model.task_type,
            "input_features": loaded_model.input_features,
            "feature_dtypes": loaded_model.feature_dtypes,
            "feature_count": loaded_model.feature_count,
            "feature_metadata_available": loaded_model.feature_metadata_available,
            "exported_at": datetime.now().isoformat()
        }

        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_config(self, loaded_model: LoadedModel, export_dir: Path) -> None:
        """Generate config.json with model section pre-filled."""
        logger.info("Generating config.json...")

        config = {
            "model": {
                "name": loaded_model.name,
                "version": loaded_model.version,
                "model_type": loaded_model.model_type
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info"
            },
            "data_source": {
                "type": "none",
                "_comment_type": "Options: 'none', 'database', 'csv', 'json'",
                "database": {
                    "connection_string": "",
                    "query": "SELECT * FROM your_table_or_view",
                    "_comment": "Example: postgresql+psycopg2://user:pass@host:5432/db"
                },
                "file": {
                    "path": "",
                    "format": "csv",
                    "_comment_format": "Options: 'csv', 'json'"
                }
            },
            "data_output": {
                "type": "response",
                "_comment_type": "Options: 'response', 'database', 'csv', 'json', 'database+response', 'csv+response', 'json+response'",
                "database": {
                    "connection_string": "",
                    "table_name": "predictions_output",
                    "if_exists": "append",
                    "_comment_if_exists": "Options: 'append', 'replace', 'fail'"
                },
                "file": {
                    "path": "",
                    "format": "csv"
                }
            }
        }

        config_path = export_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _create_zip(self, export_dir: Path, model_name: str, tmp_base: Path) -> Path:
        """Create a zip archive from the export directory."""
        logger.info("Creating zip archive...")

        zip_path = tmp_base / f"{model_name}_standalone.zip"
        root_folder = f"{model_name}_standalone"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(export_dir.rglob("*")):
                if file_path.is_file():
                    arcname = f"{root_folder}/{file_path.relative_to(export_dir).as_posix()}"
                    zf.write(file_path, arcname)

        logger.info(f"  Zip created: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        return zip_path
