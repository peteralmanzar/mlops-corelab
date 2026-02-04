"""
Shared dependencies for FastAPI routes.
"""

from fastapi import Request

from .model_manager import ModelManager


def get_model_manager(request: Request) -> ModelManager:
    """Get the model manager from app state."""
    return request.app.state.model_manager
