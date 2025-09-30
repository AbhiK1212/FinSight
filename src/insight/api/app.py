from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .routers import prediction, health
from ..core.config import get_settings
from ..monitoring.middleware import MetricsMiddleware


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="FinSight Financial Sentiment Analysis API",
        description="Financial sentiment analysis using fine-tuned DistilBERT",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.api_debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(MetricsMiddleware)
    
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])
    @app.get("/metrics")
    async def get_metrics():
        """Expose Prometheus metrics for monitoring."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    return app


app = create_app()
