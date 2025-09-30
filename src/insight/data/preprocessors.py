"""Text preprocessing utilities for financial headlines."""

import re
import string
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..core.logging import LoggerMixin


class FinancialTextPreprocessor(LoggerMixin):
    """Preprocessor for financial text data with domain-specific cleaning."""

    def __init__(self):
        """Initialize the preprocessor with financial domain patterns."""
        # Financial abbreviations and their expansions
        self.financial_abbreviations = {
            r'\bIPO\b': 'initial public offering',
            r'\bCEO\b': 'chief executive officer',
            r'\bCFO\b': 'chief financial officer',
            r'\bGDP\b': 'gross domestic product',
            r'\bP/E\b': 'price to earnings',
            r'\bROI\b': 'return on investment',
            r'\bATH\b': 'all time high',
            r'\bATL\b': 'all time low',
            r'\bYoY\b': 'year over year',
            r'\bQoQ\b': 'quarter over quarter',
            r'\bEPS\b': 'earnings per share',
            r'\bETF\b': 'exchange traded fund',
            r'\bRSI\b': 'relative strength index',
            r'\bMACD\b': 'moving average convergence divergence',
        }

        # Financial entities to preserve
        self.financial_entities = [
            r'\$[A-Z]{1,5}',  # Stock tickers like $AAPL
            r'\b[A-Z]{1,5}\b(?:\s+stock|\s+shares)?',  # Stock symbols
            r'\$[\d,]+(?:\.\d{2})?[KMB]?',  # Currency amounts
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\b\d{4}\s*Q[1-4]\b',  # Quarters like 2023 Q4
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',  # Dates
        ]

        # Market sentiment keywords
        self.sentiment_indicators = {
            'positive': [
                'surge', 'rally', 'bullish', 'gains', 'rises', 'soars', 'climbs',
                'outperforms', 'beats estimates', 'record high', 'breakthrough',
                'optimistic', 'confident', 'strong results', 'exceeds expectations'
            ],
            'negative': [
                'plunge', 'crash', 'bearish', 'losses', 'falls', 'tumbles', 'drops',
                'underperforms', 'misses estimates', 'record low', 'disappointing',
                'pessimistic', 'concerned', 'weak results', 'below expectations'
            ],
            'neutral': [
                'stable', 'unchanged', 'holds', 'maintains', 'flat', 'sideways',
                'mixed', 'cautious', 'moderate', 'steady'
            ]
        }

        # Company name variations to normalize
        self.company_normalizations = {
            r'\bApple Inc\.?\b': 'Apple',
            r'\bMicrosoft Corp\.?\b': 'Microsoft',
            r'\bAlphabet Inc\.?\b': 'Google',
            r'\bAmazon\.com Inc\.?\b': 'Amazon',
            r'\bTesla Inc\.?\b': 'Tesla',
            r'\bMeta Platforms Inc\.?\b': 'Meta',
            r'\bJPMorgan Chase & Co\.?\b': 'JPMorgan',
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize financial text."""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase but preserve financial entities
        original_text = text
        entities = self._extract_financial_entities(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Restore preserved entities
        for entity in entities:
            text = text.replace(entity.lower(), entity)

        # Expand financial abbreviations
        for abbrev, expansion in self.financial_abbreviations.items():
            text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)

        # Normalize company names
        for pattern, replacement in self.company_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities to preserve during cleaning."""
        entities = []
        for pattern in self.financial_entities:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        return entities

    def tokenize_financial_text(self, text: str) -> List[str]:
        """Tokenize text while preserving financial entities."""
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Extract financial entities
        entities = self._extract_financial_entities(cleaned_text)
        
        # Replace entities with placeholders
        entity_map = {}
        for i, entity in enumerate(entities):
            placeholder = f"__ENTITY_{i}__"
            entity_map[placeholder] = entity
            cleaned_text = cleaned_text.replace(entity, placeholder)

        # Basic tokenization (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b|[^\w\s]', cleaned_text)
        
        # Restore entities
        for i, token in enumerate(tokens):
            if token in entity_map:
                tokens[i] = entity_map[token]

        return tokens

    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment-related features from financial text."""
        text_lower = text.lower()
        features = {}

        # Count sentiment indicators
        for sentiment, keywords in self.sentiment_indicators.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{sentiment}_keywords'] = count

        # Calculate sentiment ratios
        total_sentiment_words = sum(features.values())
        if total_sentiment_words > 0:
            for sentiment in self.sentiment_indicators.keys():
                features[f'{sentiment}_ratio'] = features[f'{sentiment}_keywords'] / total_sentiment_words
        else:
            for sentiment in self.sentiment_indicators.keys():
                features[f'{sentiment}_ratio'] = 0.0

        # Additional features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return features

    def prepare_training_data(
        self,
        headlines: List[str],
        labels: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare training, validation, and test datasets."""
        # Create DataFrame
        df = pd.DataFrame({
            'headline': headlines,
            'label': labels
        })

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['headline'])
        self.logger.info(
            "Removed duplicate headlines",
            initial_size=initial_size,
            final_size=len(df),
            removed=initial_size - len(df)
        )

        # Clean headlines
        df['cleaned_headline'] = df['headline'].apply(self.clean_text)
        
        # Remove empty headlines after cleaning
        df = df[df['cleaned_headline'].str.strip() != '']

        # Extract sentiment features
        sentiment_features = df['cleaned_headline'].apply(self.extract_sentiment_features)
        feature_df = pd.DataFrame(sentiment_features.tolist())
        df = pd.concat([df, feature_df], axis=1)

        # Split into train, validation, and test sets
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label']
        )

        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_df['label']
        )

        self.logger.info(
            "Dataset split completed",
            total_size=len(df),
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            label_distribution=df['label'].value_counts().to_dict()
        )

        return train_df, val_df, test_df

    def balance_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = 'label',
        method: str = 'undersample'
    ) -> pd.DataFrame:
        """Balance the dataset using undersampling or oversampling."""
        label_counts = df[target_column].value_counts()
        self.logger.info("Original label distribution", counts=label_counts.to_dict())

        if method == 'undersample':
            # Undersample to the minority class size
            min_count = label_counts.min()
            balanced_dfs = []
            
            for label in label_counts.index:
                label_df = df[df[target_column] == label]
                sampled_df = label_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(sampled_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif method == 'oversample':
            # Oversample to the majority class size
            max_count = label_counts.max()
            balanced_dfs = []
            
            for label in label_counts.index:
                label_df = df[df[target_column] == label]
                if len(label_df) < max_count:
                    # Oversample with replacement
                    additional_samples = max_count - len(label_df)
                    oversampled = label_df.sample(n=additional_samples, replace=True, random_state=42)
                    label_df = pd.concat([label_df, oversampled], ignore_index=True)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            raise ValueError("Method must be 'undersample' or 'oversample'")

        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        new_label_counts = balanced_df[target_column].value_counts()
        self.logger.info(
            "Balanced dataset created",
            method=method,
            original_size=len(df),
            balanced_size=len(balanced_df),
            new_distribution=new_label_counts.to_dict()
        )

        return balanced_df

    def create_augmented_data(
        self,
        headlines: List[str],
        labels: List[str],
        augmentation_factor: int = 2
    ) -> Tuple[List[str], List[str]]:
        """Create augmented data through paraphrasing and synonym replacement."""
        augmented_headlines = []
        augmented_labels = []

        # Simple augmentation strategies
        for headline, label in zip(headlines, labels):
            # Original headline
            augmented_headlines.append(headline)
            augmented_labels.append(label)

            # Create variations
            for i in range(augmentation_factor - 1):
                # Synonym replacement (simplified)
                augmented_text = self._apply_synonym_replacement(headline)
                augmented_headlines.append(augmented_text)
                augmented_labels.append(label)

        self.logger.info(
            "Data augmentation completed",
            original_size=len(headlines),
            augmented_size=len(augmented_headlines),
            augmentation_factor=augmentation_factor
        )

        return augmented_headlines, augmented_labels

    def _apply_synonym_replacement(self, text: str) -> str:
        """Apply simple synonym replacement for data augmentation."""
        # Simple synonym dictionary for financial terms
        synonyms = {
            'rise': 'increase',
            'fall': 'decrease',
            'company': 'firm',
            'profit': 'earnings',
            'revenue': 'sales',
            'growth': 'expansion',
            'decline': 'drop',
            'stock': 'share',
            'market': 'trading',
            'investor': 'shareholder',
        }

        words = text.split()
        augmented_words = []

        for word in words:
            # 30% chance of replacement
            if word.lower() in synonyms and len(augmented_words) > 0:
                if hash(text) % 10 < 3:  # Simple deterministic randomness
                    augmented_words.append(synonyms[word.lower()])
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)

        return ' '.join(augmented_words)

    def validate_text_quality(self, text: str) -> Dict[str, bool]:
        """Validate text quality for training data."""
        validations = {}

        # Check minimum length
        validations['min_length'] = len(text.split()) >= 3

        # Check maximum length
        validations['max_length'] = len(text.split()) <= 100

        # Check for non-ASCII characters (financial text should be mostly ASCII)
        validations['ascii_only'] = all(ord(char) < 128 for char in text)

        # Check for excessive punctuation
        punct_ratio = sum(1 for char in text if char in string.punctuation) / len(text) if text else 0
        validations['reasonable_punctuation'] = punct_ratio < 0.3

        # Check for financial relevance (contains at least one financial keyword)
        financial_keywords = [
            'stock', 'market', 'price', 'earnings', 'revenue', 'profit', 'loss',
            'investment', 'trading', 'financial', 'economic', 'company', 'business'
        ]
        validations['financial_relevance'] = any(
            keyword in text.lower() for keyword in financial_keywords
        )

        return validations

    def get_preprocessing_stats(self, texts: List[str]) -> Dict[str, any]:
        """Get preprocessing statistics for a list of texts."""
        stats = {
            'total_texts': len(texts),
            'avg_length': sum(len(text.split()) for text in texts) / len(texts) if texts else 0,
            'min_length': min(len(text.split()) for text in texts) if texts else 0,
            'max_length': max(len(text.split()) for text in texts) if texts else 0,
            'empty_texts': sum(1 for text in texts if not text.strip()),
        }

        # Character distribution
        all_text = ' '.join(texts)
        stats['total_chars'] = len(all_text)
        stats['unique_chars'] = len(set(all_text))
        stats['avg_chars_per_text'] = len(all_text) / len(texts) if texts else 0

        # Financial entity counts
        financial_entity_counts = 0
        for text in texts:
            entities = self._extract_financial_entities(text)
            financial_entity_counts += len(entities)
        
        stats['financial_entities_total'] = financial_entity_counts
        stats['avg_financial_entities'] = financial_entity_counts / len(texts) if texts else 0

        return stats
