"""Middleware for collecting application metrics."""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start tracking request
        start_time = time.time()
        metrics.increment_active_requests()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract endpoint (remove query params and dynamic parts)
            endpoint = self._clean_endpoint(request.url.path)
            
            # Track metrics
            metrics.track_http_request(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            # Track error
            duration = time.time() - start_time
            endpoint = self._clean_endpoint(request.url.path)
            
            metrics.track_http_request(
                method=request.method,
                endpoint=endpoint,
                status=500,
                duration=duration
            )
            
            raise e
        
        finally:
            metrics.decrement_active_requests()
    
    def _clean_endpoint(self, path: str) -> str:
        """Clean endpoint path for metrics grouping."""
        # Group similar endpoints together
        if path.startswith("/api/v1/predict"):
            if path == "/api/v1/predict/batch":
                return "/api/v1/predict/batch"
            else:
                return "/api/v1/predict"
        elif path.startswith("/api/v1/health"):
            return "/api/v1/health"
        elif path == "/metrics":
            return "/metrics"
        elif path == "/docs" or path.startswith("/docs"):
            return "/docs"
        elif path == "/openapi.json":
            return "/openapi.json"
        else:
            return path
