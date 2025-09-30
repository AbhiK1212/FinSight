"""Data collectors for gathering financial headlines from various sources."""

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

from ..core.config import get_settings
import logging


class NewsArticle(BaseModel):
    """Model for a news article."""

    title: str = Field(..., description="Article headline")
    description: Optional[str] = Field(None, description="Article description")
    url: str = Field(..., description="Article URL")
    source: str = Field(..., description="Source name")
    published_at: datetime = Field(..., description="Publication timestamp")
    author: Optional[str] = Field(None, description="Article author")
    category: Optional[str] = Field(None, description="Article category")
    sentiment_label: Optional[str] = Field(None, description="Sentiment label if available")


class DataCollector(ABC):
    """Abstract base class for data collectors."""

    def __init__(self, rate_limit: int = 60):
        """Initialize collector with rate limiting."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.collected_urls: Set[str] = set()

    @abstractmethod
    async def collect_headlines(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """Collect headlines from the data source."""
        pass

    async def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = asyncio.get_event_loop().time()

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL."""
        unique_articles = []
        for article in articles:
            if article.url not in self.collected_urls:
                self.collected_urls.add(article.url)
                unique_articles.append(article)
        return unique_articles


class NewsAPICollector(DataCollector):
    """Collector for NewsAPI.org."""

    def __init__(self, api_key: str, rate_limit: int = 500):
        """Initialize NewsAPI collector."""
        super().__init__(rate_limit)
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
        # Financial keywords for targeted collection
        self.financial_keywords = [
            "stock market", "earnings", "financial", "investment",
            "trading", "economy", "GDP", "inflation", "Federal Reserve",
            "Wall Street", "NYSE", "NASDAQ", "S&P 500", "Dow Jones",
            "cryptocurrency", "bitcoin", "blockchain", "fintech",
            "merger", "acquisition", "IPO", "dividend", "quarterly results"
        ]

    async def collect_headlines(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """Collect headlines from NewsAPI."""
        await self._rate_limit()

        if not from_date:
            from_date = datetime.now() - timedelta(days=30)
        if not to_date:
            to_date = datetime.now()

        articles = []
        
        async with aiohttp.ClientSession() as session:
            # Search for general financial news
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.api_key,
                "pageSize": min(limit, 100),  # API limit
            }

            try:
                articles.extend(await self._fetch_articles(session, params))

                # Collect from specific financial sources
                financial_sources = [
                    "bloomberg", "reuters", "financial-times", "wall-street-journal",
                    "cnbc", "marketwatch", "business-insider", "fortune"
                ]

                for source in financial_sources:
                    if len(articles) >= limit:
                        break
                        
                    source_params = params.copy()
                    source_params["sources"] = source
                    source_params.pop("q", None)  # Remove query when using sources
                    
                    await self._rate_limit()
                    articles.extend(await self._fetch_articles(session, source_params))

            except Exception as e:
                self.logger.error(f"Failed to collect from NewsAPI: {str(e)}")

        articles = self._deduplicate_articles(articles)
        self.logger.info(f"Collected {len(articles)} articles from NewsAPI (query: {query}, dates: {from_date.date()} to {to_date.date()})")

        return articles[:limit]

    async def _fetch_articles(
        self, session: aiohttp.ClientSession, params: Dict
    ) -> List[NewsArticle]:
        """Fetch articles from NewsAPI endpoint."""
        url = f"{self.base_url}/everything"
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                articles = []
                
                for item in data.get("articles", []):
                    if self._is_financial_content(item.get("title", "")):
                        try:
                            article = NewsArticle(
                                title=item["title"],
                                description=item.get("description"),
                                url=item["url"],
                                source=item["source"]["name"],
                                published_at=datetime.fromisoformat(
                                    item["publishedAt"].replace("Z", "+00:00")
                                ),
                                author=item.get("author"),
                                category="financial",
                            )
                            articles.append(article)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse article: {str(e)} | Item: {str(item)[:100]}")
                
                return articles
            else:
                response_text = await response.text()
                self.logger.error(f"NewsAPI request failed: status {response.status}, response: {response_text[:200]}")
                return []

    def _is_financial_content(self, text: str) -> bool:
        """Check if text contains financial keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)


class YahooFinanceCollector(DataCollector):
    """Collector for Yahoo Finance RSS feeds and news."""

    def __init__(self, rate_limit: int = 2000):
        """Initialize Yahoo Finance collector."""
        super().__init__(rate_limit)
        self.base_url = "https://finance.yahoo.com"
        
        # RSS feed URLs for different categories
        self.rss_feeds = {
            "general": "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "markets": "https://feeds.finance.yahoo.com/rss/2.0/category-stocks",
            "economy": "https://feeds.finance.yahoo.com/rss/2.0/category-economy",
            "technology": "https://feeds.finance.yahoo.com/rss/2.0/category-technology",
        }

    async def collect_headlines(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """Collect headlines from Yahoo Finance."""
        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()

        articles = []
        
        async with aiohttp.ClientSession() as session:
            for category, feed_url in self.rss_feeds.items():
                if len(articles) >= limit:
                    break
                    
                await self._rate_limit()
                try:
                    category_articles = await self._fetch_rss_articles(
                        session, feed_url, category, from_date, to_date
                    )
                    articles.extend(category_articles)
                except Exception as e:
                    self.logger.error(
                        "Failed to collect from Yahoo Finance RSS",
                        category=category,
                        error=str(e)
                    )

        # Filter by query if provided
        if query:
            articles = [
                article for article in articles
                if query.lower() in article.title.lower()
                or (article.description and query.lower() in article.description.lower())
            ]

        articles = self._deduplicate_articles(articles)
        self.logger.info(
            "Collected articles from Yahoo Finance",
            count=len(articles),
            query=query,
            date_range=f"{from_date.date()} to {to_date.date()}"
        )

        return articles[:limit]

    async def _fetch_rss_articles(
        self,
        session: aiohttp.ClientSession,
        feed_url: str,
        category: str,
        from_date: datetime,
        to_date: datetime,
    ) -> List[NewsArticle]:
        """Fetch articles from RSS feed."""
        try:
            async with session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_rss_content(content, category, from_date, to_date)
                else:
                    self.logger.error(
                        "RSS feed request failed",
                        url=feed_url,
                        status=response.status
                    )
                    return []
        except Exception as e:
            self.logger.error("Failed to fetch RSS feed", url=feed_url, error=str(e))
            return []

    def _parse_rss_content(
        self, content: str, category: str, from_date: datetime, to_date: datetime
    ) -> List[NewsArticle]:
        """Parse RSS content to extract articles."""
        articles = []
        
        # Simple regex-based RSS parsing (in production, use feedparser)
        item_pattern = r'<item>(.*?)</item>'
        title_pattern = r'<title><!\[CDATA\[(.*?)\]\]></title>'
        link_pattern = r'<link>(.*?)</link>'
        description_pattern = r'<description><!\[CDATA\[(.*?)\]\]></description>'
        pubdate_pattern = r'<pubDate>(.*?)</pubDate>'

        items = re.findall(item_pattern, content, re.DOTALL)
        
        for item in items:
            try:
                title_match = re.search(title_pattern, item)
                link_match = re.search(link_pattern, item)
                desc_match = re.search(description_pattern, item)
                date_match = re.search(pubdate_pattern, item)

                if title_match and link_match and date_match:
                    # Parse date (simplified)
                    pub_date = datetime.now()  # Simplified for demo
                    
                    if from_date <= pub_date <= to_date:
                        article = NewsArticle(
                            title=title_match.group(1).strip(),
                            description=desc_match.group(1).strip() if desc_match else None,
                            url=link_match.group(1).strip(),
                            source="Yahoo Finance",
                            published_at=pub_date,
                            category=category,
                        )
                        articles.append(article)
                        
            except Exception as e:
                self.logger.warning("Failed to parse RSS item", error=str(e))

        return articles


class AlphaVantageCollector(DataCollector):
    """Collector for Alpha Vantage news and sentiment data."""

    def __init__(self, api_key: str, rate_limit: int = 5):
        """Initialize Alpha Vantage collector."""
        super().__init__(rate_limit)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    async def collect_headlines(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsArticle]:
        """Collect news and sentiment from Alpha Vantage."""
        await self._rate_limit()

        articles = []
        
        async with aiohttp.ClientSession() as session:
            # Collect news sentiment data
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": query if query else "AAPL,GOOGL,MSFT,AMZN,TSLA",
                "apikey": self.api_key,
                "limit": str(min(limit, 1000)),
            }

            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "feed" in data:
                            for item in data["feed"]:
                                try:
                                    # Extract sentiment score and label
                                    sentiment_score = item.get("overall_sentiment_score", 0)
                                    sentiment_label = item.get("overall_sentiment_label", "neutral")
                                    
                                    article = NewsArticle(
                                        title=item["title"],
                                        description=item.get("summary"),
                                        url=item["url"],
                                        source=item["source"],
                                        published_at=datetime.fromisoformat(
                                            item["time_published"][:8] + "T" + 
                                            item["time_published"][9:15]
                                        ),
                                        category="financial",
                                        sentiment_label=sentiment_label,
                                    )
                                    articles.append(article)
                                    
                                except Exception as e:
                                    self.logger.warning(
                                        "Failed to parse Alpha Vantage item",
                                        error=str(e),
                                        item=item
                                    )
                    else:
                        self.logger.error(
                            "Alpha Vantage request failed",
                            status=response.status,
                            response=await response.text()
                        )

            except Exception as e:
                self.logger.error("Failed to collect from Alpha Vantage", error=str(e))

        articles = self._deduplicate_articles(articles)
        self.logger.info(
            "Collected articles from Alpha Vantage",
            count=len(articles),
            query=query
        )

        return articles[:limit]


class DataCollectorFactory:
    """Factory for creating data collectors."""

    @staticmethod
    def create_collectors() -> List[DataCollector]:
        """Create all available data collectors based on configuration."""
        settings = get_settings()
        collectors = []

        data_configs = settings.get_data_source_configs()

        if "newsapi" in data_configs:
            collectors.append(
                NewsAPICollector(
                    api_key=data_configs["newsapi"]["api_key"],
                    rate_limit=data_configs["newsapi"]["rate_limit"],
                )
            )

        if "alpha_vantage" in data_configs:
            collectors.append(
                AlphaVantageCollector(
                    api_key=data_configs["alpha_vantage"]["api_key"],
                    rate_limit=data_configs["alpha_vantage"]["rate_limit"],
                )
            )

        if "yahoo_finance" in data_configs:
            collectors.append(
                YahooFinanceCollector(
                    rate_limit=data_configs["yahoo_finance"]["rate_limit"]
                )
            )

        return collectors

    @staticmethod
    async def collect_from_all_sources(
        query: str = "financial markets",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit_per_source: int = 100,
    ) -> List[NewsArticle]:
        """Collect data from all available sources."""
        collectors = DataCollectorFactory.create_collectors()
        all_articles = []

        # Collect from all sources concurrently
        tasks = [
            collector.collect_headlines(query, from_date, to_date, limit_per_source)
            for collector in collectors
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                # Log error but continue with other sources
                logger = LoggerMixin().logger
                logger.error("Collector failed", error=str(result))

        # Remove duplicates across all sources
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        return unique_articles
