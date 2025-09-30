"""DistilBERT-based financial sentiment classifier."""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import numpy as np

from ..core.config import get_settings
from ..core.logging import LoggerMixin


class FinancialSentimentDataset(Dataset):
    """Dataset class for financial sentiment data."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        labels: Optional[List[int]] = None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class FinancialDistilBertForSequenceClassification(PreTrainedModel):
    """DistilBERT model for financial sentiment classification."""

    config_class = DistilBertConfig

    def __init__(self, config: DistilBertConfig):
        """Initialize the model."""
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # DistilBERT backbone
        self.distilbert = DistilBertModel(config)
        
        # Classification head with dropout for regularization
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Additional financial domain-specific layers
        self.financial_attention = nn.MultiheadAttention(
            embed_dim=config.dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(config.dim)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        """Forward pass of the model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (batch_size, seq_len, dim)

        # Apply financial attention mechanism
        attended_output, _ = self.financial_attention(
            hidden_state, hidden_state, hidden_state,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Combine original and attended representations
        combined_output = hidden_state + attended_output
        combined_output = self.layer_norm(combined_output)

        # Pool the sequence (use [CLS] token representation)
        pooled_output = combined_output[:, 0]  # (batch_size, dim)

        # Apply pre-classifier layer
        pooled_output = self.pre_classifier(pooled_output)  # (batch_size, dim)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Final classification
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            # Calculate cross-entropy loss with class weights for imbalanced data
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


class FinancialSentimentClassifier(LoggerMixin):
    """Main classifier class for financial sentiment analysis."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """Initialize the sentiment classifier."""
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Label mappings
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.is_trained = False

        self.logger.info(
            "Initialized FinancialSentimentClassifier",
            model_name=model_name,
            num_labels=num_labels,
            device=self.device
        )

    def initialize_model(self, pretrained_path: Optional[str] = None) -> None:
        """Initialize or load the model."""
        try:
            # Initialize tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            
            if pretrained_path and os.path.exists(pretrained_path):
                # Load pre-trained model
                self.logger.info("Loading pre-trained model", path=pretrained_path)
                config = DistilBertConfig.from_pretrained(pretrained_path)
                config.num_labels = self.num_labels
                self.model = FinancialDistilBertForSequenceClassification.from_pretrained(
                    pretrained_path, config=config
                )
                self.is_trained = True
            else:
                # Initialize new model
                self.logger.info("Initializing new model")
                config = DistilBertConfig.from_pretrained(self.model_name)
                config.num_labels = self.num_labels
                config.seq_classif_dropout = 0.2  # Add dropout for regularization
                
                self.model = FinancialDistilBertForSequenceClassification.from_pretrained(
                    self.model_name, config=config
                )
                self.is_trained = False

            # Move model to device
            self.model.to(self.device)
            
            self.logger.info("Model initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize model", error=str(e))
            raise

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for better financial sentiment analysis."""
        processed_texts = []
        
        for text in texts:
            # Basic cleaning
            text = text.strip()
            
            # Financial domain-specific preprocessing
            # Normalize financial symbols
            text = self._normalize_financial_symbols(text)
            
            # Handle financial abbreviations
            text = self._expand_financial_abbreviations(text)
            
            processed_texts.append(text)
        
        return processed_texts

    def _normalize_financial_symbols(self, text: str) -> str:
        """Normalize financial symbols in text."""
        import re
        
        # Normalize stock symbols (e.g., $AAPL -> Apple stock)
        stock_patterns = {
            r'\$AAPL\b': 'Apple stock',
            r'\$GOOGL?\b': 'Google stock',
            r'\$MSFT\b': 'Microsoft stock',
            r'\$AMZN\b': 'Amazon stock',
            r'\$TSLA\b': 'Tesla stock',
            r'\$META\b': 'Meta stock',
        }
        
        for pattern, replacement in stock_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _expand_financial_abbreviations(self, text: str) -> str:
        """Expand financial abbreviations for better understanding."""
        import re
        
        abbreviations = {
            r'\bIPO\b': 'Initial Public Offering',
            r'\bP/E\b': 'Price to Earnings ratio',
            r'\bROI\b': 'Return on Investment',
            r'\bYoY\b': 'Year over Year',
            r'\bQoQ\b': 'Quarter over Quarter',
            r'\bEPS\b': 'Earnings Per Share',
        }
        
        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
        return text

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = False,
        batch_size: int = 32
    ) -> Union[str, List[str], Tuple[Union[str, List[str]], Union[float, List[float]]]]:
        """Predict sentiment for given text(s)."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)

        # Create dataset and dataloader
        dataset = FinancialSentimentDataset(
            texts=processed_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Predict
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # Get predictions
                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())

                # Get probabilities
                if return_probabilities:
                    batch_probs = F.softmax(logits, dim=-1)
                    probabilities.extend(batch_probs.cpu().numpy())

        # Convert predictions to labels
        predicted_labels = [self.id_to_label[pred] for pred in predictions]

        # Handle output format
        if single_input:
            result = predicted_labels[0]
            if return_probabilities:
                return result, probabilities[0]
            return result
        else:
            if return_probabilities:
                return predicted_labels, probabilities
            return predicted_labels

    def predict_with_confidence(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Predict sentiment with confidence scores for each class."""
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        predictions, probabilities = self.predict(
            texts, return_probabilities=True, batch_size=batch_size
        )

        # Format results with confidence scores
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'predicted_sentiment': pred,
                'confidence_scores': {
                    label: float(probs[label_id])
                    for label, label_id in self.label_to_id.items()
                },
                'max_confidence': float(max(probs))
            }
            results.append(result)

        return results[0] if single_input else results

    def save_model(self, save_path: str) -> None:
        """Save the trained model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No model to save. Initialize model first.")

        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'is_trained': self.is_trained,
        }
        
        import json
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info("Model saved successfully", path=save_path)

    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.num_labels = metadata.get('num_labels', self.num_labels)
            self.max_length = metadata.get('max_length', self.max_length)
            self.label_to_id = metadata.get('label_to_id', self.label_to_id)
            self.id_to_label = metadata.get('id_to_label', self.id_to_label)
            self.is_trained = metadata.get('is_trained', True)

        # Initialize model from saved weights
        self.initialize_model(model_path)
        
        self.logger.info("Model loaded successfully", path=model_path)

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        if not self.model:
            return {"status": "not_initialized"}

        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "device": self.device,
            "is_trained": self.is_trained,
            "label_mapping": self.label_to_id,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
