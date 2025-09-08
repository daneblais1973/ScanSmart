from pydantic import BaseModel, validator
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Enums ---
class CatalystType(str, Enum):
    EARNINGS = "Earnings / Guidance"
    MA = "M&A / Partnerships"
    REGULATORY = "Regulatory / FDA"
    FDA_APPROVAL = "FDA Approval"
    MERGER_ACQUISITION = "Merger / Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    PARTNERSHIP = "Partnership"
    INSIDER_TRADING = "Insider Trading"
    ANALYST_UPGRADE = "Analyst Upgrade"
    NEWS = "News"
    ANALYST = "Analyst Update"
    INSIDER = "Insider Trading"
    GENERAL = "General News"

class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

class SourceType(str, Enum):
    NEWSAPI = "NewsAPI"
    TWITTER = "Twitter"
    REDDIT = "Reddit"
    RSS = "RSS"
    REGULATORY = "Regulatory"
    FINANCIAL = "Financial"
    MANUAL = "Manual"
    UNKNOWN = "Unknown"

# --- Core Models ---
class StockInfo(BaseModel):
    ticker: str
    sector: str = "Unknown"
    longName: str = ""
    marketCap: Optional[float] = None
    industry: Optional[str] = None

class NewsItem(BaseModel):
    title: str
    content: Optional[str] = None
    source: SourceType = SourceType.UNKNOWN
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    author: Optional[str] = None
    source_name: Optional[str] = None

class Catalyst(BaseModel):
    ticker: str
    catalyst: str
    category: CatalystType = CatalystType.GENERAL
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    sector: str = "Unknown"
    sentiment_score: float = 0.0
    impact: int = 0
    source: SourceType = SourceType.UNKNOWN
    confidence: float = 0.0
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = {}

    @validator('impact')
    def validate_impact(cls, v):
        return max(0, min(100, v))

    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))

    @validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        return max(-1.0, min(1.0, v))

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

# --- Request/Response Models ---
class CatalystRequest(BaseModel):
    tickers: List[str]
    min_impact_score: int = 70
    sources: List[SourceType] = [SourceType.NEWSAPI, SourceType.TWITTER, SourceType.REDDIT]
    max_results: int = 10
    filters: Optional[Dict[str, Any]] = None

class CatalystResponse(BaseModel):
    catalysts: List[Catalyst]
    total_count: int
    processing_time: float
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessTextRequest(BaseModel):
    text: str
    stock_info: Optional[StockInfo] = None
    source: SourceType = SourceType.UNKNOWN

class ProcessTextResponse(BaseModel):
    category: CatalystType
    sentiment_score: float
    sentiment_label: SentimentLabel
    is_relevant: bool = True
    confidence: float = 0.0
    entities: List[Dict[str, Any]] = []

# --- API Configuration Models ---
class APICredentials(BaseModel):
    newsapi_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "QuantumCatalystScanner/1.0"

class ServiceConfig(BaseModel):
    min_impact_score: int = 70
    max_tickers: int = 20
    refresh_interval: int = 300  # seconds
    enabled_sources: List[SourceType] = [SourceType.NEWSAPI, SourceType.TWITTER, SourceType.REDDIT]

# --- Alert Models ---
class AlertRule(BaseModel):
    min_impact_score: int = 70
    tickers: Optional[List[str]] = None
    categories: Optional[List[CatalystType]] = None
    enabled: bool = True

class NotificationConfig(BaseModel):
    type: str  # email, sms, webhook, console
    config: Dict[str, Any]
    enabled: bool = True

class AlertRequest(BaseModel):
    catalyst: Catalyst
    rules: Optional[AlertRule] = None

class AlertResponse(BaseModel):
    alert_id: str
    sent: bool
    results: Dict[str, Any]
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- Health & Status Models ---
class HealthStatus(BaseModel):
    status: str
    timestamp: datetime = datetime.now()
    service: str
    version: str
    details: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ServiceStatus(BaseModel):
    service: str
    status: str
    uptime: Optional[float] = None
    last_check: datetime = datetime.now()
    dependencies: List[Dict[str, Any]] = []

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- Data Ingestion Models ---
class DataFetchRequest(BaseModel):
    ticker: str
    limit: int = 20
    sources: List[SourceType] = [SourceType.NEWSAPI, SourceType.TWITTER, SourceType.REDDIT]

class DataFetchResponse(BaseModel):
    ticker: str
    news: List[NewsItem] = []
    twitter: List[NewsItem] = []
    reddit: List[NewsItem] = []
    total_results: int = 0
    processing_time: float = 0.0
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- Error Models ---
class ErrorResponse(BaseModel):
    error: str
    code: int
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- Utility Functions for Models ---
def catalyst_to_dict(catalyst: Catalyst) -> Dict[str, Any]:
    """Convert Catalyst model to dictionary for JSON serialization"""
    return catalyst.dict()

def dict_to_catalyst(data: Dict[str, Any]) -> Catalyst:
    """Convert dictionary to Catalyst model"""
    return Catalyst(**data)

def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker format"""
    if not ticker or len(ticker) > 10:
        return False
    return ticker.replace('.', '').replace('-', '').isalpha()

def generate_catalyst_id(ticker: str, timestamp: datetime = None) -> str:
    """Generate unique ID for catalyst"""
    if timestamp is None:
        timestamp = datetime.now()
    return f"{ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
