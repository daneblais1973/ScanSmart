import logging
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path
import threading

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType
from core.config import AppConfig

logger = logging.getLogger(__name__)

Base = declarative_base()

class CatalystRecord(Base):
    """SQLAlchemy model for catalyst records"""
    __tablename__ = 'catalysts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    catalyst = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    sentiment_label = Column(String(20), nullable=False)
    sector = Column(String(100), default='Unknown')
    sentiment_score = Column(Float, default=0.0)
    impact = Column(Integer, default=0)
    source = Column(String(20), nullable=False)
    confidence = Column(Float, default=0.0)
    published_date = Column(DateTime(timezone=True), nullable=True)
    created_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    url = Column(Text, nullable=True)
    extra_data = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)

class AlertRecord(Base):
    """SQLAlchemy model for alert records"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    catalyst_id = Column(Integer, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    recipient = Column(String(200), nullable=False)
    sent_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), default='sent')
    extra_data = Column(JSON, default=dict)

class FetcherStatsRecord(Base):
    """SQLAlchemy model for fetcher statistics"""
    __tablename__ = 'fetcher_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fetcher_name = Column(String(50), nullable=False, index=True)
    date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)

class TradingSignalRecord(Base):
    """SQLAlchemy model for trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # day_trading, momentum, long_term
    signal_strength = Column(Float, nullable=False)
    direction = Column(Integer, nullable=False)  # -1, 0, 1
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    risk_level = Column(String(10), default='MEDIUM')
    catalyst_id = Column(Integer, nullable=True, index=True)
    technical_score = Column(Float, default=0.0)
    fundamental_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    created_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_date = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default='active')  # active, expired, triggered, cancelled
    performance = Column(JSON, default=dict)  # Track actual vs predicted performance
    extra_metadata = Column(JSON, default=dict)

class TechnicalAnalysisRecord(Base):
    """SQLAlchemy model for technical analysis data"""
    __tablename__ = 'technical_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    analysis_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    indicators = Column(JSON, default=dict)  # All technical indicators
    patterns = Column(JSON, default=dict)  # Detected patterns
    trading_signals = Column(JSON, default=dict)  # Signals for different trading types
    risk_metrics = Column(JSON, default=dict)  # Risk assessment
    technical_score = Column(JSON, default=dict)  # Overall technical scores
    data_quality = Column(String(10), default='medium')

class FundamentalAnalysisRecord(Base):
    """SQLAlchemy model for fundamental analysis data"""
    __tablename__ = 'fundamental_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    analysis_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    sector = Column(String(100), nullable=True)
    market_cap = Column(Float, nullable=True)
    ratios = Column(JSON, default=dict)  # Financial ratios
    growth_metrics = Column(JSON, default=dict)  # Growth analysis
    valuation_metrics = Column(JSON, default=dict)  # Valuation analysis
    health_metrics = Column(JSON, default=dict)  # Financial health
    efficiency_metrics = Column(JSON, default=dict)  # Operational efficiency
    dividend_metrics = Column(JSON, default=dict)  # Dividend analysis
    fundamental_score = Column(JSON, default=dict)  # Overall fundamental scores
    investment_signals = Column(JSON, default=dict)  # Investment recommendations
    data_quality = Column(String(10), default='medium')

class CatalystCorrelationRecord(Base):
    """SQLAlchemy model for catalyst correlation tracking"""
    __tablename__ = 'catalyst_correlations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    catalyst_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    catalyst_category = Column(String(50), nullable=False)
    historical_impact = Column(Float, default=0.0)  # Historical price impact
    correlation_score = Column(Float, default=0.0)  # Correlation with similar catalysts
    sector_correlation = Column(Float, default=0.0)  # Sector-wide impact correlation
    success_rate = Column(Float, default=0.0)  # Success rate of similar catalysts
    time_to_impact = Column(Integer, default=0)  # Hours until price impact
    volatility_increase = Column(Float, default=0.0)  # Volatility increase
    volume_increase = Column(Float, default=0.0)  # Volume increase
    options_activity = Column(JSON, default=dict)  # Options flow data
    created_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class PerformanceTrackingRecord(Base):
    """SQLAlchemy model for tracking signal performance"""
    __tablename__ = 'performance_tracking'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)
    entry_date = Column(DateTime(timezone=True), nullable=False)
    exit_date = Column(DateTime(timezone=True), nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    predicted_direction = Column(Integer, nullable=False)  # -1, 0, 1
    actual_direction = Column(Integer, nullable=True)  # -1, 0, 1
    predicted_return = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)
    max_gain = Column(Float, default=0.0)
    max_loss = Column(Float, default=0.0)
    holding_period_hours = Column(Integer, nullable=True)
    success = Column(Boolean, nullable=True)
    confidence_at_entry = Column(Float, nullable=False)
    catalyst_impact_realized = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    extra_metadata = Column(JSON, default=dict)

class DatabaseManager:
    """Database manager for catalyst scanner"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._engine = None
        self._SessionLocal = None
        self._lock = threading.Lock()
        
        self._init_database()
        
    def _init_database(self):
        """Initialize database connection and create tables"""
        try:
            # Create engine
            if self.config.database.url.startswith('postgresql://'):
                # PostgreSQL
                self._engine = create_engine(
                    self.config.database.url,
                    pool_size=self.config.database.pool_size,
                    max_overflow=self.config.database.max_overflow,
                    echo=self.config.debug
                )
            else:
                # SQLite fallback
                db_path = self.config.database.url.replace('sqlite:///', '')
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self._engine = create_engine(
                    self.config.database.url,
                    echo=self.config.debug,
                    connect_args={"check_same_thread": False}  # For SQLite
                )
            
            # Create session factory
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            
            # Create tables
            Base.metadata.create_all(bind=self._engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with proper cleanup"""
        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def is_connected(self) -> bool:
        """Check if database connection is alive"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def save_catalyst(self, catalyst: Catalyst) -> Optional[int]:
        """Save catalyst to database"""
        try:
            with self.get_session() as session:
                # Check for existing catalyst to avoid duplicates
                existing = session.query(CatalystRecord).filter(
                    CatalystRecord.ticker == catalyst.ticker,
                    CatalystRecord.catalyst == catalyst.catalyst,
                    CatalystRecord.source == (catalyst.source.value if hasattr(catalyst.source, 'value') else str(catalyst.source))
                ).first()
                
                if existing:
                    logger.debug(f"Catalyst already exists: {catalyst.ticker} - {catalyst.catalyst[:50]}...")
                    return int(existing.id)
                
                # Create new record
                record = CatalystRecord(
                    ticker=catalyst.ticker,
                    catalyst=catalyst.catalyst,
                    category=catalyst.category.value if hasattr(catalyst.category, 'value') else str(catalyst.category),
                    sentiment_label=catalyst.sentiment_label.value if hasattr(catalyst.sentiment_label, 'value') else str(catalyst.sentiment_label),
                    sector=catalyst.sector,
                    sentiment_score=catalyst.sentiment_score,
                    impact=catalyst.impact,
                    source=catalyst.source.value if hasattr(catalyst.source, 'value') else str(catalyst.source),
                    confidence=catalyst.confidence,
                    published_date=catalyst.published_date,
                    url=catalyst.url,
                    extra_data=catalyst.metadata
                )
                
                session.add(record)
                session.flush()  # Get the ID without committing
                
                logger.info(f"Saved catalyst: {catalyst.ticker} - {catalyst.catalyst[:50]}...")
                session.commit()
                return record.id
                
        except Exception as e:
            logger.error(f"Failed to save catalyst: {e}")
            return None
    
    def get_catalysts(self, 
                     ticker: Optional[str] = None,
                     category: Optional[str] = None,
                     sentiment: Optional[str] = None,
                     min_impact: Optional[int] = None,
                     limit: int = 100) -> List[CatalystRecord]:
        """Get catalysts with filtering"""
        try:
            with self.get_session() as session:
                query = session.query(CatalystRecord).filter(CatalystRecord.is_active == True)
                
                if ticker:
                    query = query.filter(CatalystRecord.ticker == ticker)
                if category:
                    query = query.filter(CatalystRecord.category == category)
                if sentiment:
                    query = query.filter(CatalystRecord.sentiment_label == sentiment)
                if min_impact is not None:
                    query = query.filter(CatalystRecord.impact >= min_impact)
                
                return query.order_by(CatalystRecord.created_date.desc()).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to get catalysts: {e}")
            return []
    
    def get_catalyst_count(self) -> int:
        """Get total count of active catalysts"""
        try:
            with self.get_session() as session:
                return session.query(CatalystRecord).filter(CatalystRecord.is_active == True).count()
        except Exception as e:
            logger.error(f"Failed to get catalyst count: {e}")
            return 0
    
    def clear_old_catalysts(self, days: int = 30) -> int:
        """Clear catalysts older than specified days"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                deleted_count = session.query(CatalystRecord).filter(
                    CatalystRecord.created_date < cutoff_date
                ).delete()
                
                session.commit()
                logger.info(f"Cleared {deleted_count} old catalysts")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to clear old catalysts: {e}")
            return 0
    
    def save_trading_signal(self, signal_data: Dict[str, Any]) -> Optional[int]:
        """Save trading signal to database"""
        try:
            with self.get_session() as session:
                record = TradingSignalRecord(**signal_data)
                session.add(record)
                session.flush()
                return record.id
        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
            return None
    
    def save_technical_analysis(self, analysis_data: Dict[str, Any]) -> Optional[int]:
        """Save technical analysis to database"""
        try:
            with self.get_session() as session:
                record = TechnicalAnalysisRecord(**analysis_data)
                session.add(record)
                session.flush()
                return record.id
        except Exception as e:
            logger.error(f"Failed to save technical analysis: {e}")
            return None
    
    def save_fundamental_analysis(self, analysis_data: Dict[str, Any]) -> Optional[int]:
        """Save fundamental analysis to database"""
        try:
            with self.get_session() as session:
                record = FundamentalAnalysisRecord(**analysis_data)
                session.add(record)
                session.flush()
                return record.id
        except Exception as e:
            logger.error(f"Failed to save fundamental analysis: {e}")
            return None
    
    def track_performance(self, performance_data: Dict[str, Any]) -> Optional[int]:
        """Track signal performance"""
        try:
            with self.get_session() as session:
                record = PerformanceTrackingRecord(**performance_data)
                session.add(record)
                session.flush()
                return record.id
        except Exception as e:
            logger.error(f"Failed to track performance: {e}")
            return None