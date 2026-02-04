"""
FastAPI Model Serving Application

Main entry point for the model serving service.
Loads all champion models from MLflow at startup and serves predictions via REST API.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .model_manager import ModelManager
from .routes import health, predict, admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads all champion models on startup and cleans up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("Model Serving API - Starting up")
    logger.info("=" * 60)

    # Get configuration from environment
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    champion_alias = os.environ.get("CHAMPION_ALIAS", "champion")

    logger.info(f"MLflow Tracking URI: {tracking_uri}")
    logger.info(f"Champion Alias: {champion_alias}")

    # Initialize model manager and store in app state
    model_manager = ModelManager(
        tracking_uri=tracking_uri,
        champion_alias=champion_alias
    )
    app.state.model_manager = model_manager

    # Load all champion models
    try:
        loaded = model_manager.load_all_champions()
        if loaded:
            logger.info(f"Successfully loaded {len(loaded)} champion model(s):")
            for name, m in loaded.items():
                logger.info(f"  - {name} v{m.version} ({m.model_type})")
        else:
            logger.warning("No champion models found. Service will start but predictions will fail.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("Service starting with no models loaded. Use /admin/reload to retry.")

    logger.info("=" * 60)
    logger.info("Model Serving API - Ready")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Model Serving API - Shutting down")
    model_manager.models.clear()


# Create FastAPI application
app = FastAPI(
    title="MLOps CoreLab Model Serving",
    description="""
    FastAPI service for serving ML models from MLflow Model Registry.

    ## Features

    - **Automatic model discovery**: Loads all champion models from MLflow on startup
    - **Multi-model serving**: Each model is available at `/predict/{model_name}`
    - **Hot-reload**: Use `/admin/reload` to reload models after promotion
    - **Health checks**: `/health` and `/health/ready` endpoints for orchestration

    ## Usage

    1. **List available models**: `GET /admin/models`
    2. **Make predictions**: `POST /predict/{model_name}` with JSON body
    3. **Reload after promotion**: `POST /admin/reload`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(admin.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc)
        }
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "service": "MLOps CoreLab Model Serving",
        "docs": "/docs",
        "health": "/health",
        "models": "/admin/models"
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
