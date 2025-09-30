#!/usr/bin/env python3
"""
Database Setup Script for FinSight
Creates necessary tables and initializes the database
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker
from insight.core.config import get_settings

Base = declarative_base()

class Article(Base):
    """Financial news articles table."""
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)
    source = Column(String(100), nullable=False)
    published_at = Column(DateTime, nullable=False)
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Prediction(Base):
    """Model predictions table."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    predicted_sentiment = Column(String(20), nullable=False)
    confidence_score = Column(Float)
    model_version = Column(String(50))
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    """Model performance metrics table."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_split = Column(String(20))  # train, validation, test
    created_at = Column(DateTime, default=datetime.utcnow)

def setup_database():
    """Create database tables and test connection."""
    print("🗄️  Setting up FinSight database...")
    
    # Get database configuration
    settings = get_settings()
    print(f"Connecting to: {settings.database_url}")
    
    try:
        # Create engine
        engine = create_engine(settings.database_url, echo=False)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ Database connection successful!")
        
        # Create all tables
        print("📋 Creating database tables...")
        Base.metadata.create_all(engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Insert test record
        test_article = Article(
            title="Test Article: Apple reports strong earnings",
            url="https://example.com/test-article",
            source="Test Source",
            published_at=datetime.utcnow(),
            sentiment_label="positive",
            confidence_score=0.95
        )
        
        session.add(test_article)
        session.commit()
        
        # Verify tables were created
        article_count = session.query(Article).count()
        prediction_count = session.query(Prediction).count()
        metrics_count = session.query(ModelMetrics).count()
        
        print(f"✅ Database setup completed!")
        print(f"   Articles table: {article_count} records")
        print(f"   Predictions table: {prediction_count} records") 
        print(f"   Model metrics table: {metrics_count} records")
        
        # Test a simple query
        latest_article = session.query(Article).order_by(Article.created_at.desc()).first()
        if latest_article:
            print(f"   Latest article: '{latest_article.title[:50]}...'")
        
        session.close()
        
        print("\n🎉 Database is ready for FinSight!")
        print("   - Articles table: Stores financial news")
        print("   - Predictions table: Stores model predictions")
        print("   - Model metrics table: Stores performance metrics")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check PostgreSQL container is running: docker ps | grep postgres")
        print("2. Check database URL in .env file")
        print("3. Verify port 5433 is not blocked")
        return False
    
    return True

def test_database_operations():
    """Test basic database operations."""
    print("\n🧪 Testing database operations...")
    
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Test insert
        new_article = Article(
            title="Tesla stock surges on strong delivery numbers",
            url="https://example.com/tesla-news",
            source="Financial News",
            published_at=datetime.utcnow(),
            sentiment_label="positive",
            confidence_score=0.88
        )
        
        session.add(new_article)
        session.commit()
        
        # Test query
        articles = session.query(Article).filter(Article.sentiment_label == "positive").all()
        print(f"✅ Found {len(articles)} positive articles")
        
        # Test prediction insert
        new_prediction = Prediction(
            text="Apple beats earnings expectations",
            predicted_sentiment="positive",
            confidence_score=0.92,
            model_version="v1.0",
            processing_time_ms=45.2
        )
        
        session.add(new_prediction)
        session.commit()
        
        predictions = session.query(Prediction).all()
        print(f"✅ Found {len(predictions)} predictions")
        
        session.close()
        print("✅ All database operations working correctly!")
        
    except Exception as e:
        print(f"❌ Database operations failed: {e}")
        session.rollback()
        session.close()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 FinSight Database Setup")
    print("=" * 40)
    
    # Setup database
    if setup_database():
        # Test operations
        if test_database_operations():
            print("\n🎉 Database setup and testing completed successfully!")
            print("\nNext steps:")
            print("1. Start Redis: docker run -d --name finsight-redis -p 6379:6379 redis:7-alpine")
            print("2. Test API: uvicorn src.insight.api.app:app --reload")
            print("3. View docs: http://localhost:8000/docs")
        else:
            print("\n❌ Database operations test failed")
            sys.exit(1)
    else:
        print("\n❌ Database setup failed")
        sys.exit(1)
