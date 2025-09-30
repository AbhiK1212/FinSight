from datetime import datetime
from fastapi import APIRouter

from ..models import HealthResponse
from ...models.model_serving import get_model_server
from ...models.caching import get_cache
from ...core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        model_server = get_model_server()
        model_info = model_server.get_model_info()
        model_loaded = model_info.get("status") == "loaded"
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_info=model_info,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/ready")
async def readiness_check():
    try:
        model_server = get_model_server()
        model_info = model_server.get_model_info()
        
        if model_info.get("status") != "loaded":
            return {"status": "not_ready"}, 503
        
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}, 503


@router.get("/live")
async def liveness_check():
    return {"status": "alive"}


@router.get("/cache")
async def cache_status():
    """Get cache performance statistics."""
    try:
        cache = get_cache()
        stats = cache.get_stats()
        return {
            "cache_status": "enabled" if stats.get("enabled") else "disabled",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache status failed: {e}")
        return {
            "cache_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
