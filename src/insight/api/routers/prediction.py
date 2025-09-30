import time
from fastapi import APIRouter, HTTPException

from ..models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from ...models.model_serving import get_model_server
from ...monitoring.metrics import metrics
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    start_time = time.time()
    
    try:
        model_server = get_model_server()
        
        if request.return_confidence:
            result = model_server.predict_with_confidence(request.text)
            response = PredictionResponse(
                sentiment=result["predicted_sentiment"],
                confidence=result["max_confidence"],
                confidence_scores=result["confidence_scores"],
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Track metrics with confidence
            metrics.track_prediction(
                sentiment=response.sentiment,
                confidence=response.confidence,
                duration=response.processing_time_ms / 1000,
                prediction_type="single"
            )
        else:
            sentiment = model_server.predict(request.text)
            response = PredictionResponse(
                sentiment=sentiment,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Track metrics without confidence
            metrics.track_prediction(
                sentiment=response.sentiment,
                confidence=None,
                duration=response.processing_time_ms / 1000,
                prediction_type="single"
            )
        
        logger.info(f"Prediction completed: {response.sentiment} (took {response.processing_time_ms:.2f}ms)")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)} | Text: {request.text[:100]}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict/batch", response_model=BatchPredictionResponse)  
async def predict_sentiment_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    start_time = time.time()
    
    try:
        model_server = get_model_server()
        
        if request.return_confidence:
            results = model_server.predict_with_confidence(request.texts)
            predictions = [
                PredictionResponse(
                    sentiment=result["predicted_sentiment"],
                    confidence=result["max_confidence"],
                    confidence_scores=result["confidence_scores"],
                    processing_time_ms=0
                )
                for result in results
            ]
        else:
            sentiments = model_server.predict(request.texts)
            predictions = [
                PredictionResponse(sentiment=sentiment, processing_time_ms=0)
                for sentiment in sentiments
            ]
        
        total_time = time.time() - start_time
        
        # Track batch metrics
        metrics.track_batch_prediction(
            batch_size_count=len(request.texts),
            total_duration=total_time
        )
        
        # Track individual predictions if confidence is available
        if request.return_confidence:
            for prediction in predictions:
                metrics.track_prediction(
                    sentiment=prediction.sentiment,
                    confidence=prediction.confidence,
                    duration=0,  # Individual duration not tracked in batch
                    prediction_type="batch"
                )
        else:
            for prediction in predictions:
                metrics.track_prediction(
                    sentiment=prediction.sentiment,
                    confidence=None,
                    duration=0,
                    prediction_type="batch"
                )
        
        logger.info(f"Batch prediction completed: {len(request.texts)} texts (took {total_time * 1000:.2f}ms)")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time * 1000
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)} | Batch size: {len(request.texts)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
