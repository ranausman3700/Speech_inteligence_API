"""FastAPI HTTP and WebSocket entrypoints."""

from speech_intelligence_api.entrypoints.http.app import create_app

__all__ = ["create_app"]
