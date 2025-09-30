from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    return_confidence: bool = Field(default=False)


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    return_confidence: bool = Field(default=False)


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: Optional[float] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None
    timestamp: str
