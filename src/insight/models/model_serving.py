import time
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from ..core.config import get_settings
from .caching import get_cache
import logging

logger = logging.getLogger(__name__)


class ModelServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None):
        settings = get_settings()
        model_path = model_path or settings.model_path
        
        try:
            # Try loading fine-tuned model first
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
                local_files_only=True
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded fine-tuned model from {model_path}")
            
        except (OSError, ValueError):
            # Fallback to base model for development
            logger.warning("Fine-tuned model not found, using base model")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=3
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
        logger.info(f"Model server ready on device: {self.device}")
    
    def predict(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_loaded:
            self.load_model()
            
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        # Convert to labels
        predicted_labels = [self.label_map[pred.item()] for pred in predictions]
        
        return predicted_labels[0] if single_input else predicted_labels
    
    def predict_with_confidence(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if not self.is_loaded:
            self.load_model()
        
        # Handle single text input with caching
        if isinstance(text, str):
            cache = get_cache()
            
            # Try cache first
            cached_result = cache.get(text)
            if cached_result:
                return cached_result
        
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        inputs = self.tokenizer(
            texts,
            padding=True, 
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        results = []
        for pred, probs in zip(predictions, probabilities):
            result = {
                "predicted_sentiment": self.label_map[pred.item()],
                "confidence_scores": {
                    label: float(probs[idx]) 
                    for idx, label in self.label_map.items()
                },
                "max_confidence": float(torch.max(probs))
            }
            results.append(result)
        
        # Cache single predictions
        if single_input and isinstance(text, str):
            cache = get_cache()
            cache.set(text, results[0])
            
        return results[0] if single_input else results
    
    def get_model_info(self) -> Dict:
        return {
            "status": "loaded" if self.is_loaded else "not_loaded",
            "device": self.device,
            "model_type": "DistilBERT",
            "num_labels": 3,
            "labels": list(self.label_map.values())
        }


# Global model server instance
_model_server = None

def get_model_server() -> ModelServer:
    global _model_server
    if _model_server is None:
        _model_server = ModelServer()
    return _model_server
