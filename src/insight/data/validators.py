"""Data validation utilities for ensuring data quality in the pipeline."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel, Field, validator

from ..core.logging import LoggerMixin
from .collectors import NewsArticle


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment."""
    
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    duplicate_records: int = Field(..., description="Number of duplicate records")
    completeness_score: float = Field(..., description="Data completeness score (0-1)")
    validity_score: float = Field(..., description="Data validity score (0-1)")
    uniqueness_score: float = Field(..., description="Data uniqueness score (0-1)")
    overall_quality_score: float = Field(..., description="Overall quality score (0-1)")
    issues: List[str] = Field(default_factory=list, description="List of quality issues")


class ValidationRule(BaseModel):
    """Individual validation rule."""
    
    rule_name: str = Field(..., description="Name of the validation rule")
    description: str = Field(..., description="Description of what the rule validates")
    severity: str = Field(..., description="Severity level: ERROR, WARNING, INFO")
    is_required: bool = Field(default=True, description="Whether this rule is required")
    
    @validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        if v not in ["ERROR", "WARNING", "INFO"]:
            raise ValueError("Severity must be ERROR, WARNING, or INFO")
        return v


class DataValidator(LoggerMixin):
    """Comprehensive data validator for financial headlines."""

    def __init__(self):
        """Initialize the data validator with validation rules."""
        self.validation_rules = self._setup_validation_rules()
        self.financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'earnings', 'revenue',
            'profit', 'loss', 'financial', 'economic', 'company', 'business',
            'shares', 'dividend', 'merger', 'acquisition', 'ipo', 'cryptocurrency',
            'bitcoin', 'blockchain', 'fintech', 'banking', 'insurance', 'nasdaq',
            'nyse', 'sp500', 'dow', 'russell', 'volatility', 'liquidity'
        ]

    def _setup_validation_rules(self) -> List[ValidationRule]:
        """Set up validation rules for financial headlines."""
        return [
            ValidationRule(
                rule_name="title_required",
                description="Title field must be present and non-empty",
                severity="ERROR",
                is_required=True
            ),
            ValidationRule(
                rule_name="title_length",
                description="Title must be between 10 and 200 characters",
                severity="WARNING",
                is_required=False
            ),
            ValidationRule(
                rule_name="url_format",
                description="URL must be a valid HTTP/HTTPS URL",
                severity="ERROR",
                is_required=True
            ),
            ValidationRule(
                rule_name="source_required",
                description="Source field must be present and non-empty",
                severity="ERROR",
                is_required=True
            ),
            ValidationRule(
                rule_name="published_date",
                description="Published date must be valid and within reasonable range",
                severity="ERROR",
                is_required=True
            ),
            ValidationRule(
                rule_name="financial_relevance",
                description="Content should be financially relevant",
                severity="WARNING",
                is_required=False
            ),
            ValidationRule(
                rule_name="duplicate_detection",
                description="Articles should not be duplicates",
                severity="WARNING",
                is_required=False
            ),
            ValidationRule(
                rule_name="text_quality",
                description="Text should meet quality standards",
                severity="WARNING",
                is_required=False
            ),
        ]

    def validate_article(self, article: NewsArticle) -> Tuple[bool, List[str]]:
        """Validate a single news article."""
        is_valid = True
        issues = []

        # Title validation
        if not article.title or not article.title.strip():
            is_valid = False
            issues.append("Title is empty or missing")
        elif len(article.title) < 10:
            issues.append("Title is too short (< 10 characters)")
        elif len(article.title) > 200:
            issues.append("Title is too long (> 200 characters)")

        # URL validation
        if not self._is_valid_url(article.url):
            is_valid = False
            issues.append("Invalid URL format")

        # Source validation
        if not article.source or not article.source.strip():
            is_valid = False
            issues.append("Source is empty or missing")

        # Date validation
        if not self._is_valid_date(article.published_at):
            is_valid = False
            issues.append("Invalid or unreasonable publication date")

        # Financial relevance
        if not self._is_financially_relevant(article.title + " " + (article.description or "")):
            issues.append("Content may not be financially relevant")

        # Text quality
        quality_issues = self._check_text_quality(article.title)
        issues.extend(quality_issues)

        return is_valid, issues

    def validate_articles_batch(self, articles: List[NewsArticle]) -> DataQualityMetrics:
        """Validate a batch of articles and return quality metrics."""
        total_records = len(articles)
        valid_records = 0
        all_issues = []
        
        # Track duplicates
        seen_urls = set()
        duplicate_count = 0

        validation_results = []

        for article in articles:
            # Check for duplicates
            if article.url in seen_urls:
                duplicate_count += 1
                all_issues.append(f"Duplicate URL: {article.url}")
            else:
                seen_urls.add(article.url)

            # Validate individual article
            is_valid, issues = self.validate_article(article)
            validation_results.append((is_valid, issues))
            
            if is_valid:
                valid_records += 1
            
            all_issues.extend(issues)

        # Calculate quality metrics
        invalid_records = total_records - valid_records
        completeness_score = self._calculate_completeness_score(articles)
        validity_score = valid_records / total_records if total_records > 0 else 0
        uniqueness_score = (total_records - duplicate_count) / total_records if total_records > 0 else 0
        
        # Overall quality score (weighted average)
        overall_quality_score = (
            0.4 * validity_score +
            0.3 * completeness_score +
            0.3 * uniqueness_score
        )

        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_count,
            completeness_score=completeness_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            overall_quality_score=overall_quality_score,
            issues=list(set(all_issues))  # Remove duplicate issues
        )

        self.logger.info(
            "Batch validation completed",
            total_records=total_records,
            valid_records=valid_records,
            quality_score=overall_quality_score,
            issues_count=len(metrics.issues)
        )

        return metrics

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Validate a pandas DataFrame containing article data."""
        required_columns = ['title', 'url', 'source', 'published_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to NewsArticle objects for validation
        articles = []
        for _, row in df.iterrows():
            try:
                article = NewsArticle(
                    title=row.get('title', ''),
                    description=row.get('description'),
                    url=row.get('url', ''),
                    source=row.get('source', ''),
                    published_at=pd.to_datetime(row['published_at']),
                    author=row.get('author'),
                    category=row.get('category'),
                    sentiment_label=row.get('sentiment_label')
                )
                articles.append(article)
            except Exception as e:
                self.logger.warning(f"Failed to parse row: {e}", row=row.to_dict())

        return self.validate_articles_batch(articles)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))

    def _is_valid_date(self, date: datetime) -> bool:
        """Check if date is valid and reasonable."""
        if not isinstance(date, datetime):
            return False
        
        # Date should not be in the future
        if date > datetime.now():
            return False
        
        # Date should not be too old (more than 10 years)
        ten_years_ago = datetime.now() - timedelta(days=3650)
        if date < ten_years_ago:
            return False
        
        return True

    def _is_financially_relevant(self, text: str) -> bool:
        """Check if text is financially relevant."""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)

    def _check_text_quality(self, text: str) -> List[str]:
        """Check text quality and return issues."""
        issues = []
        
        if not text:
            return ["Text is empty"]
        
        # Check for excessive capitalization
        if sum(1 for c in text if c.isupper()) / len(text) > 0.5:
            issues.append("Excessive capitalization")
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!@#$%^&*()') / len(text)
        if punct_ratio > 0.1:
            issues.append("Excessive special characters")
        
        # Check for reasonable word count
        word_count = len(text.split())
        if word_count < 3:
            issues.append("Too few words")
        elif word_count > 50:
            issues.append("Too many words for a headline")
        
        # Check for suspicious patterns
        if re.search(r'(.)\1{4,}', text):  # Repeated characters
            issues.append("Contains repeated characters")
        
        if text.count('?') > 2:
            issues.append("Too many question marks")
        
        if text.count('!') > 2:
            issues.append("Too many exclamation marks")
        
        return issues

    def _calculate_completeness_score(self, articles: List[NewsArticle]) -> float:
        """Calculate data completeness score."""
        if not articles:
            return 0.0
        
        total_fields = 0
        filled_fields = 0
        
        for article in articles:
            # Count required fields
            required_fields = ['title', 'url', 'source', 'published_at']
            for field in required_fields:
                total_fields += 1
                if getattr(article, field, None):
                    filled_fields += 1
            
            # Count optional fields
            optional_fields = ['description', 'author', 'category']
            for field in optional_fields:
                total_fields += 1
                if getattr(article, field, None):
                    filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 0.0

    def clean_invalid_articles(
        self, 
        articles: List[NewsArticle], 
        remove_invalid: bool = True
    ) -> List[NewsArticle]:
        """Clean articles by removing or fixing invalid ones."""
        cleaned_articles = []
        removed_count = 0
        
        for article in articles:
            is_valid, issues = self.validate_article(article)
            
            if is_valid or not remove_invalid:
                # Try to fix minor issues
                fixed_article = self._fix_article_issues(article, issues)
                cleaned_articles.append(fixed_article)
            else:
                removed_count += 1
                self.logger.debug(
                    "Removed invalid article",
                    title=article.title[:50] + "..." if len(article.title) > 50 else article.title,
                    issues=issues
                )
        
        self.logger.info(
            "Article cleaning completed",
            original_count=len(articles),
            cleaned_count=len(cleaned_articles),
            removed_count=removed_count
        )
        
        return cleaned_articles

    def _fix_article_issues(self, article: NewsArticle, issues: List[str]) -> NewsArticle:
        """Attempt to fix minor issues in articles."""
        # Create a copy to avoid modifying the original
        article_dict = article.dict()
        
        # Fix title issues
        if "Title is too short" in str(issues) and article.description:
            # Extend title with description if available
            article_dict['title'] = f"{article.title} - {article.description[:50]}"
        
        # Normalize source names
        source_normalizations = {
            'reuters.com': 'Reuters',
            'bloomberg.com': 'Bloomberg',
            'cnbc.com': 'CNBC',
            'wsj.com': 'Wall Street Journal',
            'ft.com': 'Financial Times',
        }
        
        for domain, normalized_name in source_normalizations.items():
            if domain in article.url.lower():
                article_dict['source'] = normalized_name
                break
        
        return NewsArticle(**article_dict)

    def generate_validation_report(self, metrics: DataQualityMetrics) -> str:
        """Generate a human-readable validation report."""
        report = f"""
Data Quality Validation Report
==============================

Overall Summary:
- Total Records: {metrics.total_records}
- Valid Records: {metrics.valid_records}
- Invalid Records: {metrics.invalid_records}
- Duplicate Records: {metrics.duplicate_records}

Quality Scores:
- Completeness: {metrics.completeness_score:.2%}
- Validity: {metrics.validity_score:.2%}
- Uniqueness: {metrics.uniqueness_score:.2%}
- Overall Quality: {metrics.overall_quality_score:.2%}

Quality Assessment:
"""
        
        if metrics.overall_quality_score >= 0.9:
            report += "✅ EXCELLENT - Data quality is very high\n"
        elif metrics.overall_quality_score >= 0.8:
            report += "✅ GOOD - Data quality is acceptable\n"
        elif metrics.overall_quality_score >= 0.7:
            report += "⚠️ FAIR - Data quality needs improvement\n"
        else:
            report += "❌ POOR - Data quality is unacceptable\n"

        if metrics.issues:
            report += f"\nIssues Found ({len(metrics.issues)}):\n"
            for i, issue in enumerate(metrics.issues[:10], 1):  # Show first 10 issues
                report += f"{i}. {issue}\n"
            
            if len(metrics.issues) > 10:
                report += f"... and {len(metrics.issues) - 10} more issues\n"

        return report

    def create_data_profiling_report(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Create a comprehensive data profiling report."""
        if not articles:
            return {"error": "No articles provided"}

        # Basic statistics
        total_articles = len(articles)
        sources = {}
        categories = {}
        dates = []
        title_lengths = []
        description_lengths = []

        for article in articles:
            # Source distribution
            sources[article.source] = sources.get(article.source, 0) + 1
            
            # Category distribution
            if article.category:
                categories[article.category] = categories.get(article.category, 0) + 1
            
            # Date range
            dates.append(article.published_at)
            
            # Text length statistics
            title_lengths.append(len(article.title))
            if article.description:
                description_lengths.append(len(article.description))

        # Calculate statistics
        dates.sort()
        date_range = {
            "earliest": dates[0].isoformat() if dates else None,
            "latest": dates[-1].isoformat() if dates else None,
            "span_days": (dates[-1] - dates[0]).days if len(dates) > 1 else 0
        }

        profile = {
            "summary": {
                "total_articles": total_articles,
                "unique_sources": len(sources),
                "unique_categories": len(categories),
                "date_range": date_range
            },
            "sources": dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)),
            "categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            "text_statistics": {
                "title_length": {
                    "min": min(title_lengths) if title_lengths else 0,
                    "max": max(title_lengths) if title_lengths else 0,
                    "avg": sum(title_lengths) / len(title_lengths) if title_lengths else 0
                },
                "description_length": {
                    "min": min(description_lengths) if description_lengths else 0,
                    "max": max(description_lengths) if description_lengths else 0,
                    "avg": sum(description_lengths) / len(description_lengths) if description_lengths else 0
                }
            }
        }

        return profile
