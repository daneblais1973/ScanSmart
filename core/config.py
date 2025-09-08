import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessingMode(str, Enum):
    AI = "ai"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"

class InferenceMode(str, Enum):
    LOCAL = "local"
    API = "api"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///catalyst_scanner.db'))
    host: str = field(default_factory=lambda: os.getenv('PGHOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('PGPORT', '5432')))
    database: str = field(default_factory=lambda: os.getenv('PGDATABASE', 'catalyst_scanner'))
    user: str = field(default_factory=lambda: os.getenv('PGUSER', 'postgres'))
    password: str = field(default_factory=lambda: os.getenv('PGPASSWORD', ''))
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    default_ttl: int = 300
    max_memory: str = '100mb'
    enabled: bool = True

@dataclass
class NLPConfig:
    """NLP processing configuration"""
    model_cache_dir: str = './models'
    finbert_model: str = 'ProsusAI/finbert'
    bart_model: str = 'facebook/bart-large-cnn'
    spacy_model: str = 'en_core_web_sm'
    batch_size: int = 64
    max_sequence_length: int = 512
    device: str = 'auto'
    
    # Memory optimization settings
    lazy_loading: bool = True
    model_type: str = 'heavy'
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    inference_mode: InferenceMode = InferenceMode.LOCAL
    auto_cleanup: bool = True
    memory_threshold_mb: int = 4096
    
    # Enhanced sentiment settings
    sentiment_threshold: float = 0.6
    mode: str = 'ai'
    
    # Cross-validation settings
    min_sources_for_validation: int = 2
    validation_confidence_threshold: float = 0.8
    
    # Model variants for lightweight mode
    lightweight_models: dict = field(default_factory=lambda: {
        'finbert': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'bart': 'sshleifer/distilbart-cnn-12-6',
        'spacy': 'en_core_web_sm'
    })

@dataclass
class AlertConfig:
    """Alerting configuration"""
    min_impact_score: int = 70
    min_confidence: float = 0.7
    rate_limit_minutes: int = 5
    max_alerts_per_hour: int = 20
    enabled_channels: List[str] = field(default_factory=lambda: ['email', 'console'])

@dataclass
class APIConfig:
    """API credentials and configuration - COMPLETE VERSION"""
    # Financial Data APIs
    fmp_key: str = field(default_factory=lambda: os.getenv('FMP_API_KEY', ''))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_API_KEY', ''))
    polygon_key: str = field(default_factory=lambda: os.getenv('POLYGON_API_KEY', ''))
    tiingo_key: str = field(default_factory=lambda: os.getenv('TIINGO_API_KEY', ''))
    yahoo_finance_key: str = field(default_factory=lambda: os.getenv('YAHOO_FINANCE_KEY', ''))
    
    # News & Social APIs
    newsapi_key: str = field(default_factory=lambda: os.getenv('NEWSAPI_API_KEY', ''))
    twitter_bearer_token: str = field(default_factory=lambda: os.getenv('TWITTER_BEARER_TOKEN', ''))
    reddit_client_id: str = field(default_factory=lambda: os.getenv('REDDIT_CLIENT_ID', ''))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv('REDDIT_CLIENT_SECRET', ''))
    
    # Regulatory APIs
    sec_edgar_email: str = field(default_factory=lambda: os.getenv('SEC_EDGAR_EMAIL', ''))
    fda_api_key: str = field(default_factory=lambda: os.getenv('FDA_API_KEY', ''))
    
    # AI Service APIs
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    huggingface_token: str = field(default_factory=lambda: os.getenv('HUGGINGFACE_TOKEN', ''))
    
    # Notification service credentials
    twilio_account_sid: str = field(default_factory=lambda: os.getenv('TWILIO_ACCOUNT_SID', ''))
    twilio_auth_token: str = field(default_factory=lambda: os.getenv('TWILIO_AUTH_TOKEN', ''))
    twilio_phone_number: str = field(default_factory=lambda: os.getenv('TWILIO_PHONE_NUMBER', ''))
    
    # Email/SMTP configuration
    smtp_server: str = field(default_factory=lambda: os.getenv('SMTP_SERVER', ''))
    smtp_port: int = field(default_factory=lambda: int(os.getenv('SMTP_PORT', '587')))
    smtp_username: str = field(default_factory=lambda: os.getenv('SMTP_USERNAME', ''))
    smtp_password: str = field(default_factory=lambda: os.getenv('SMTP_PASSWORD', ''))
    
    # API rate limits and configuration
    request_timeout: int = field(default_factory=lambda: int(os.getenv('API_REQUEST_TIMEOUT', '30')))
    max_retries: int = field(default_factory=lambda: int(os.getenv('API_MAX_RETRIES', '3')))
    rate_limit_delay: float = field(default_factory=lambda: float(os.getenv('API_RATE_LIMIT_DELAY', '0.5')))

@dataclass
class FetcherConfig:
    """Data fetcher configuration"""
    max_concurrent: int = 25
    timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_delay: float = 0.5
    batch_size: int = 200
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    # Enhanced parallel processing settings
    worker_pool_size: int = 15
    concurrent_nlp_workers: int = 10
    cross_validation_enabled: bool = True
    sector_specific_detection: bool = True
    dynamic_parameters: bool = True

@dataclass
class UISettings:
    """UI-specific settings"""
    theme: str = field(default_factory=lambda: os.getenv('UI_THEME', 'dark'))
    auto_refresh: bool = field(default_factory=lambda: os.getenv('UI_AUTO_REFRESH', 'true').lower() == 'true')
    refresh_interval: int = field(default_factory=lambda: int(os.getenv('UI_REFRESH_INTERVAL', '60')))
    show_notifications: bool = field(default_factory=lambda: os.getenv('UI_SHOW_NOTIFICATIONS', 'true').lower() == 'true')
    
    # Catalyst detection thresholds
    min_impact_score: int = field(default_factory=lambda: int(os.getenv('UI_MIN_IMPACT_SCORE', '50')))
    min_confidence: float = field(default_factory=lambda: float(os.getenv('UI_MIN_CONFIDENCE', '0.7')))
    max_age_hours: int = field(default_factory=lambda: int(os.getenv('UI_MAX_AGE_HOURS', '24')))
    
    # Market filters
    enabled_exchanges: List[str] = field(default_factory=lambda: os.getenv('UI_ENABLED_EXCHANGES', 'NYSE,NASDAQ').split(','))
    market_cap_min: int = field(default_factory=lambda: int(os.getenv('UI_MARKET_CAP_MIN', '100')))
    market_cap_max: int = field(default_factory=lambda: int(os.getenv('UI_MARKET_CAP_MAX', '1000')))
    volume_min: int = field(default_factory=lambda: int(os.getenv('UI_VOLUME_MIN', '100000')))

class AppConfig:
    """Main application configuration with complete API support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'config.json'
        
        # Initialize configuration sections with complete API support
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.nlp = NLPConfig()
        self.alerts = AlertConfig()
        self.api = APIConfig()  # Now includes all financial API keys
        self.fetchers = FetcherConfig()
        self.ui = UISettings()
        
        # Application settings
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.version = '1.0.0'
        
        # Load additional config from file if exists
        self.load_config_file()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded successfully with complete API support")
    
    def load_config_file(self):
        """Load configuration from JSON file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration sections
                sections = [
                    'database', 'cache', 'nlp', 'alerts', 'api', 
                    'fetchers', 'ui'
                ]
                
                for section in sections:
                    if section in config_data:
                        self._update_dataclass(getattr(self, section), config_data[section])
                
                logger.info(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        else:
            # Create default config file if it doesn't exist
            self.save_config_file()
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle special cases for enum types
                if key == 'processing_mode' and isinstance(value, str):
                    value = ProcessingMode(value)
                elif key == 'inference_mode' and isinstance(value, str):
                    value = InferenceMode(value)
                
                setattr(obj, key, value)
    
    def _validate_config(self):
        """Validate configuration settings"""
        warnings = []
        
        # Check if any financial APIs are configured
        financial_apis_configured = any([
            self.api.fmp_key,
            self.api.alpha_vantage_key,
            self.api.polygon_key,
            self.api.tiingo_key
        ])
        
        if not financial_apis_configured:
            warnings.append("No financial data API keys configured - market data will be limited")
        
        # Check if any news/social APIs are configured
        news_apis_configured = any([
            self.api.newsapi_key,
            self.api.twitter_bearer_token,
            (self.api.reddit_client_id and self.api.reddit_client_secret)
        ])
        
        if not news_apis_configured:
            warnings.append("No news/social API keys configured - catalyst detection will be limited")
        
        # Validate database configuration
        if not self.database.url and not all([self.database.host, self.database.database]):
            raise ValueError("Database configuration incomplete")
        
        # Validate NLP configuration
        if not Path(self.nlp.model_cache_dir).exists():
            Path(self.nlp.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
    
    def get_configured_sources(self) -> Dict[str, bool]:
        """Get dictionary of configured data sources and their status"""
        return {
            # Financial data sources
            'financial_modeling_prep': bool(self.api.fmp_key),
            'alpha_vantage': bool(self.api.alpha_vantage_key),
            'polygon': bool(self.api.polygon_key),
            'tiingo': bool(self.api.tiingo_key),
            
            # News & social sources
            'newsapi': bool(self.api.newsapi_key),
            'twitter': bool(self.api.twitter_bearer_token),
            'reddit': bool(self.api.reddit_client_id and self.api.reddit_client_secret),
            
            # Regulatory sources
            'sec_edgar': bool(self.api.sec_edgar_email),
            'fda': bool(self.api.fda_api_key),
            
            # AI services
            'openai': bool(self.api.openai_api_key),
            'anthropic': bool(self.api.anthropic_api_key),
            'huggingface': bool(self.api.huggingface_token),
            
            # Always available
            'rss': True
        }
    
    def save_config_file(self):
        """Save current configuration to file"""
        config_data = {
            'database': self.database.__dict__,
            'cache': self.cache.__dict__,
            'nlp': self.nlp.__dict__,
            'alerts': self.alerts.__dict__,
            'api': self.api.__dict__,
            'fetchers': self.fetchers.__dict__,
            'ui': self.ui.__dict__
        }
        
        try:
            # Create directory if it doesn't exist
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """Update a configuration section with new values"""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            self._update_dataclass(section_obj, updates)
            logger.info(f"Updated {section} configuration")
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section as dictionary"""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            return section_obj.__dict__
        else:
            raise ValueError(f"Unknown configuration section: {section}")