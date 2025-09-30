"""Prometheus metrics for FinSight application."""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
import time

# Request metrics
http_requests_total = Counter(
    'finsight_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'finsight_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Prediction metrics
predictions_total = Counter(
    'finsight_predictions_total',
    'Total number of predictions made',
    ['sentiment', 'model_version']
)

prediction_duration_seconds = Histogram(
    'finsight_prediction_duration_seconds',
    'Time spent on predictions',
    ['prediction_type']  # single or batch
)

prediction_confidence = Histogram(
    'finsight_prediction_confidence',
    'Distribution of prediction confidence scores',
    ['sentiment']
)

# Cache metrics
cache_hits_total = Counter(
    'finsight_cache_hits_total',
    'Total number of cache hits'
)

cache_misses_total = Counter(
    'finsight_cache_misses_total',
    'Total number of cache misses'
)

# Model metrics
model_load_time_seconds = Gauge(
    'finsight_model_load_time_seconds',
    'Time taken to load the model'
)

model_info = Info(
    'finsight_model_info',
    'Information about the loaded model'
)

# Business metrics
batch_size = Histogram(
    'finsight_batch_size',
    'Distribution of batch prediction sizes'
)

# System metrics
active_requests = Gauge(
    'finsight_active_requests',
    'Number of requests currently being processed'
)


class MetricsCollector:
    """Helper class for collecting metrics throughout the application."""
    
    @staticmethod
    def track_prediction(sentiment: str, confidence: float, duration: float, 
                        prediction_type: str = "single", model_version: str = "v1"):
        """Track a prediction event."""
        predictions_total.labels(
            sentiment=sentiment, 
            model_version=model_version
        ).inc()
        
        prediction_duration_seconds.labels(
            prediction_type=prediction_type
        ).observe(duration)
        
        if confidence is not None:
            prediction_confidence.labels(sentiment=sentiment).observe(confidence)
    
    @staticmethod
    def track_batch_prediction(batch_size_count: int, total_duration: float):
        """Track batch prediction metrics."""
        batch_size.observe(batch_size_count)
        prediction_duration_seconds.labels(prediction_type="batch").observe(total_duration)
    
    @staticmethod
    def track_cache_hit():
        """Track cache hit."""
        cache_hits_total.inc()
    
    @staticmethod
    def track_cache_miss():
        """Track cache miss."""
        cache_misses_total.inc()
    
    @staticmethod
    def track_http_request(method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics."""
        http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status)
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    @staticmethod
    def set_model_info(model_data: Dict[str, Any]):
        """Set model information."""
        model_info.info(model_data)
    
    @staticmethod
    def track_model_load_time(load_time: float):
        """Track model loading time."""
        model_load_time_seconds.set(load_time)
    
    @staticmethod
    def increment_active_requests():
        """Increment active request count."""
        active_requests.inc()
    
    @staticmethod
    def decrement_active_requests():
        """Decrement active request count."""
        active_requests.dec()


# Global metrics collector instance
metrics = MetricsCollector()
