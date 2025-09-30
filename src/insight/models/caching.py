"""
Prediction Caching with Redis
Improves API performance for repeated requests
"""

import json
import hashlib
import redis
import logging
from typing import Optional, Dict, Any
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class PredictionCache:
    """Redis-based caching for model predictions."""
    
    def __init__(self):
        self.settings = get_settings()
        try:
            self.redis_client = redis.from_url(self.settings.redis_url)
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("Redis cache enabled")
        except Exception as e:
            logger.warning(f"Redis cache disabled: {e}")
            self.enabled = False
            self.redis_client = None
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key from input text."""
        # Normalize text for consistent caching
        normalized_text = text.lower().strip()
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
        return f"prediction:{text_hash}"
    
    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for text."""
        if not self.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(text)
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                result = json.loads(cached_result)
                logger.info(f"Cache hit for prediction")
                return result
                
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            
        return None
    
    def set(self, text: str, prediction_result: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache prediction result with TTL."""
        if not self.enabled:
            return False
            
        try:
            cache_key = self._generate_cache_key(text)
            
            # Add cache metadata
            cache_data = {
                **prediction_result,
                "cached": True,
                "cache_key": cache_key[:8]  # First 8 chars for debugging
            }
            
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data)
            )
            
            logger.info(f"Cached prediction (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def clear_all(self) -> int:
        """Clear all cached predictions."""
        if not self.enabled:
            return 0
            
        try:
            keys = self.redis_client.keys("prediction:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cached predictions")
                return deleted
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            info = self.redis_client.info()
            keys = len(self.redis_client.keys("prediction:*"))
            
            return {
                "enabled": True,
                "total_predictions_cached": keys,
                "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}


# Global cache instance
_cache_instance = None

def get_cache() -> PredictionCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PredictionCache()
    return _cache_instance
