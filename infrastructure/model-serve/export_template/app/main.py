"""
Standalone Model Serving Application

Self-contained FastAPI service for serving a single ML model.
No MLflow dependency — model is loaded from disk at startup.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config
from .model_loader import load_model_from_disk
from .routes import health, predict, batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model on startup."""
    logger.info("=" * 60)
    logger.info("Standalone Model Serving - Starting up")
    logger.info("=" * 60)

    # Load configuration
    try:
        config = load_config()
        app.state.config = config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        app.state.config = None

    # Load model from disk
    try:
        model, metadata = load_model_from_disk()
        app.state.model = model
        app.state.metadata = metadata
        logger.info(f"Model ready: {metadata.get('name', 'unknown')} v{metadata.get('version', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.state.model = None
        app.state.metadata = {}

    logger.info("=" * 60)
    logger.info("Standalone Model Serving - Ready")
    logger.info("=" * 60)

    yield

    logger.info("Standalone Model Serving - Shutting down")


# Create FastAPI application
app = FastAPI(
    title="Standalone Model Serving",
    description="Self-contained FastAPI service for serving a single ML model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)}
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Standalone Model Serving",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch": "/predict/batch"
    }
