import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import jwt

from core.config import AppConfig
from core.database import DatabaseManager
from nlp.processor import NLPProcessor
from alerting.alert_manager import AlertManager
from ml_training.model_trainer import ModelTrainer
from backtesting.backtester import HistoricalValidator
from data_fetchers import get_all_fetchers
from shared.models import Catalyst, SentimentLabel, CatalystType

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class CatalystRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    text: str = Field(..., description="Text to analyze for catalysts")
    source: str = Field(default="api", description="Source of the information")

class CatalystResponse(BaseModel):
    id: Optional[str] = None
    ticker: str
    catalyst: str
    category: str
    sentiment_label: str
    sentiment_score: float
    impact: int
    confidence: float
    source: str
    published_date: Optional[datetime] = None
    created_date: datetime

class AnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_categories: bool = Field(default=True, description="Include category detection")

class AnalysisResponse(BaseModel):
    text: str
    sentiment: Optional[Dict[str, Any]] = None
    categories: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    processing_time: float

class AlertRequest(BaseModel):
    catalyst_id: str = Field(..., description="Catalyst ID to alert on")
    channels: List[str] = Field(..., description="Alert channels: email, sms, webhook")
    recipient: Optional[str] = Field(None, description="Recipient (email/phone)")
    message: Optional[str] = Field(None, description="Custom alert message")

class BacktestRequest(BaseModel):
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    min_impact_score: int = Field(default=50, description="Minimum impact score filter")

class ApiConfig(BaseModel):
    newsapi_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    openai_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

# Security
security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token for API authentication"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

class CatalystScannerAPI:
    """RESTful API for Financial Catalyst Scanner"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.app = FastAPI(
            title="Financial Catalyst Scanner API",
            description="RESTful API for financial catalyst detection and analysis",
            version="1.0.0"
        )
        
        # Initialize components
        self.db_manager = DatabaseManager(config)
        self.nlp_processor = NLPProcessor(config)
        self.alert_manager = AlertManager(config)
        self.model_trainer = ModelTrainer(config)
        self.historical_validator = HistoricalValidator(config)
        self.data_fetchers = get_all_fetchers(config)
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        logger.info("Catalyst Scanner API initialized")
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Financial Catalyst Scanner API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check database connection
                db_status = self.db_manager.is_connected()
                
                # Check data fetchers
                active_sources = sum(1 for f in self.data_fetchers.values() if f.is_configured())
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "database_connected": db_status,
                    "active_data_sources": active_sources,
                    "total_data_sources": len(self.data_fetchers)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
        
        @self.app.post("/analyze/text", response_model=AnalysisResponse)
        async def analyze_text(request: AnalysisRequest, token: dict = Depends(verify_token)):
            """Analyze text for sentiment, categories, and entities"""
            try:
                start_time = datetime.now()
                
                result = {
                    "text": request.text,
                    "sentiment": None,
                    "categories": None,
                    "entities": None
                }
                
                if request.include_sentiment:
                    sentiment = await self.nlp_processor.sentiment_analyzer.analyze_sentiment(request.text)
                    result["sentiment"] = sentiment
                
                if request.include_categories:
                    # Simplified category detection
                    categories = []
                    text_lower = request.text.lower()
                    if any(term in text_lower for term in ['earnings', 'revenue', 'profit']):
                        categories.append('Earnings / Guidance')
                    if any(term in text_lower for term in ['merger', 'acquisition', 'partnership']):
                        categories.append('M&A / Partnerships')
                    if any(term in text_lower for term in ['fda', 'regulatory', 'approval']):
                        categories.append('Regulatory / FDA')
                    
                    result["categories"] = categories
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                result["processing_time"] = processing_time
                
                return AnalysisResponse(**result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/catalysts", response_model=CatalystResponse)
        async def create_catalyst(request: CatalystRequest, token: dict = Depends(verify_token)):
            """Create a new catalyst from text analysis"""
            try:
                # Analyze the text
                sentiment = await self.nlp_processor.sentiment_analyzer.analyze_sentiment(request.text)
                
                # Create catalyst object
                catalyst_data = {
                    'ticker': request.ticker.upper(),
                    'catalyst': request.text[:200],  # Truncate for storage
                    'category': 'General News',
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'impact': min(100, max(10, int(abs(sentiment['score']) * 100))),
                    'confidence': sentiment['confidence'],
                    'source': request.source,
                    'published_date': datetime.now(timezone.utc),
                    'created_date': datetime.now(timezone.utc),
                    'url': None,
                    'extra_data': {'method': 'api_submission'}
                }
                
                # Store in database
                catalyst_id = self.db_manager.store_catalyst(catalyst_data)
                catalyst_data['id'] = catalyst_id
                
                return CatalystResponse(**catalyst_data)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Catalyst creation failed: {str(e)}")
        
        @self.app.get("/catalysts", response_model=List[CatalystResponse])
        async def get_catalysts(
            ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
            limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
            offset: int = Query(0, ge=0, description="Offset for pagination"),
            min_impact: int = Query(0, ge=0, le=100, description="Minimum impact score"),
            token: dict = Depends(verify_token)
        ):
            """Get catalysts with optional filtering"""
            try:
                catalysts = self.db_manager.get_recent_catalysts(
                    limit=limit,
                    ticker_filter=ticker,
                    min_impact=min_impact
                )
                
                results = []
                for catalyst in catalysts:
                    results.append(CatalystResponse(
                        id=str(catalyst.id) if hasattr(catalyst, 'id') else None,
                        ticker=catalyst.ticker,
                        catalyst=catalyst.catalyst,
                        category=catalyst.category,
                        sentiment_label=catalyst.sentiment_label.value if hasattr(catalyst.sentiment_label, 'value') else str(catalyst.sentiment_label),
                        sentiment_score=catalyst.sentiment_score,
                        impact=catalyst.impact,
                        confidence=catalyst.confidence,
                        source=catalyst.source,
                        published_date=catalyst.published_date,
                        created_date=catalyst.created_date
                    ))
                
                return results
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve catalysts: {str(e)}")
        
        @self.app.post("/alerts")
        async def send_alert(request: AlertRequest, background_tasks: BackgroundTasks, token: dict = Depends(verify_token)):
            """Send alert for a specific catalyst"""
            try:
                # Get catalyst details
                catalyst = self.db_manager.get_catalyst_by_id(request.catalyst_id)
                if not catalyst:
                    raise HTTPException(status_code=404, detail="Catalyst not found")
                
                # Prepare alert
                alert_data = {
                    'catalyst': catalyst,
                    'channels': request.channels,
                    'recipient': request.recipient,
                    'custom_message': request.message
                }
                
                # Send alert in background
                background_tasks.add_task(self._send_alert_background, alert_data)
                
                return {"message": "Alert queued for delivery", "catalyst_id": request.catalyst_id}
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Alert failed: {str(e)}")
        
        @self.app.post("/backtest")
        async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks, token: dict = Depends(verify_token)):
            """Run backtest analysis"""
            try:
                # Queue backtest in background
                background_tasks.add_task(
                    self._run_backtest_background,
                    request.start_date,
                    request.end_date,
                    request.min_impact_score
                )
                
                return {
                    "message": "Backtest analysis started",
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
        
        @self.app.post("/training/retrain")
        async def trigger_retraining(background_tasks: BackgroundTasks, token: dict = Depends(verify_token)):
            """Trigger model retraining"""
            try:
                background_tasks.add_task(self._retrain_models_background)
                return {"message": "Model retraining initiated"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
        
        @self.app.get("/training/status")
        async def get_training_status(token: dict = Depends(verify_token)):
            """Get model training status and performance history"""
            try:
                status = self.model_trainer.get_model_status()
                performance_history = self.model_trainer.get_performance_history()
                
                return {
                    "model_status": status,
                    "performance_history": performance_history[-10:]  # Last 10 records
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")
        
        @self.app.put("/config/api-keys")
        async def update_api_config(config_update: ApiConfig, token: dict = Depends(verify_token)):
            """Update API configuration"""
            try:
                # Update configuration
                api_updates = {}
                if config_update.newsapi_key:
                    api_updates['newsapi_key'] = config_update.newsapi_key
                if config_update.twitter_bearer_token:
                    api_updates['twitter_bearer_token'] = config_update.twitter_bearer_token
                if config_update.reddit_client_id:
                    api_updates['reddit_client_id'] = config_update.reddit_client_id
                if config_update.reddit_client_secret:
                    api_updates['reddit_client_secret'] = config_update.reddit_client_secret
                if config_update.openai_api_key:
                    api_updates['openai_api_key'] = config_update.openai_api_key
                if config_update.twilio_account_sid:
                    api_updates['twilio_account_sid'] = config_update.twilio_account_sid
                if config_update.twilio_auth_token:
                    api_updates['twilio_auth_token'] = config_update.twilio_auth_token
                if config_update.twilio_phone_number:
                    api_updates['twilio_phone_number'] = config_update.twilio_phone_number
                
                # Apply updates
                self.config.update_api_config(api_updates)
                
                return {"message": f"Updated {len(api_updates)} API configuration settings"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")
        
        @self.app.get("/stats")
        async def get_system_stats(token: dict = Depends(verify_token)):
            """Get system statistics"""
            try:
                # Get catalyst counts
                total_catalysts = self.db_manager.get_catalyst_count()
                
                # Get data source status
                active_sources = sum(1 for f in self.data_fetchers.values() if f.is_configured())
                
                # Get recent activity
                recent_catalysts = self.db_manager.get_recent_catalysts(limit=10)
                
                return {
                    "total_catalysts": total_catalysts,
                    "active_data_sources": active_sources,
                    "total_data_sources": len(self.data_fetchers),
                    "recent_activity": len(recent_catalysts),
                    "system_uptime": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")
    
    async def _send_alert_background(self, alert_data: Dict[str, Any]):
        """Background task for sending alerts"""
        try:
            await self.alert_manager.send_alert(
                alert_data['catalyst'],
                alert_data['channels'],
                alert_data.get('recipient'),
                alert_data.get('custom_message')
            )
            logger.info(f"Alert sent for catalyst {alert_data['catalyst'].ticker}")
        except Exception as e:
            logger.error(f"Background alert failed: {e}")
    
    async def _run_backtest_background(self, start_date: datetime, end_date: datetime, min_impact: int):
        """Background task for running backtests"""
        try:
            self.historical_validator.backtest_config.min_impact_score = min_impact
            result = await self.historical_validator.run_backtest(start_date, end_date)
            logger.info(f"Backtest completed - F1 Score: {result.f1_score:.3f}")
        except Exception as e:
            logger.error(f"Background backtest failed: {e}")
    
    async def _retrain_models_background(self):
        """Background task for model retraining"""
        try:
            result = await self.model_trainer.automated_retraining_cycle()
            logger.info(f"Retraining completed - {result['models_retrained']} models retrained")
        except Exception as e:
            logger.error(f"Background retraining failed: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)