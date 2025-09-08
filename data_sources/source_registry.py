import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    FREE_NO_ACCOUNT = "free_no_account"
    FREE_ACCOUNT_NEEDED = "free_account_needed"
    PAID_ACCOUNT_NEEDED = "paid_account_needed"

class PollingStrategy(Enum):
    FREQUENT = "frequent"  # 15-30 min
    MODERATE = "moderate"  # 30-60 min
    LIMITED = "limited"    # 60-120 min
    MANUAL = "manual"      # Manual only

@dataclass
class RateLimit:
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    concurrent_connections: Optional[int] = None
    custom_limit: Optional[str] = None

@dataclass
class DataSource:
    name: str
    url: str
    category: str
    access_level: AccessLevel
    polling_strategy: PollingStrategy
    rate_limit: RateLimit
    description: str
    api_key_required: bool = False
    authentication_method: Optional[str] = None
    terms_url: Optional[str] = None
    documentation_url: Optional[str] = None
    status: str = "active"  # active, inactive, deprecated
    tags: List[str] = field(default_factory=list)

class DataSourceRegistry:
    """Comprehensive registry of all available data sources"""
    
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all data sources from the provided lists"""
        
        # Financial APIs - Free (Account Required)
        financial_apis = [
            ("Alpha Vantage", "https://www.alphavantage.co/", "Stock, forex, and crypto data API", 500, 5),
            ("Finnhub", "https://finnhub.io/", "Real-time stocks, company fundamentals", None, 60),
            ("Financial Modeling Prep", "https://site.financialmodelingprep.com/", "Financial statements and ratios", 250, None),
            ("IEX Cloud", "https://iexcloud.io/", "US stock market data", None, None),
            ("Marketstack", "https://marketstack.com/", "End-of-day stock data", None, None),
            ("Quandl Free Tier", "https://data.nasdaq.com/", "Economic datasets", 20, None),
            ("Kaggle API", "https://www.kaggle.com/", "Financial datasets", 10, None),
            ("GitHub API", "https://api.github.com/", "Fintech repositories", None, None)
        ]
        
        for name, url, desc, daily_limit, minute_limit in financial_apis:
            rate_limit = RateLimit()
            if daily_limit:
                rate_limit.requests_per_day = daily_limit
            if minute_limit:
                rate_limit.requests_per_minute = minute_limit
            
            self.add_source(DataSource(
                name=name,
                url=url,
                category="Financial APIs",
                access_level=AccessLevel.FREE_ACCOUNT_NEEDED,
                polling_strategy=PollingStrategy.LIMITED,
                rate_limit=rate_limit,
                description=desc,
                api_key_required=True,
                authentication_method="API Key"
            ))
        
        
        self.add_source(DataSource(
            name="CoinGecko API",
            url="https://www.coingecko.com/en/api",
            category="Cryptocurrency",
            access_level=AccessLevel.FREE_NO_ACCOUNT,
            polling_strategy=PollingStrategy.MODERATE,
            rate_limit=RateLimit(requests_per_minute=30),
            description="Crypto prices and market data",
            api_key_required=False,
            documentation_url="https://www.coingecko.com/en/api/documentation"
        ))
        
        # News APIs
        self.add_source(DataSource(
            name="NewsAPI Free",
            url="https://newsapi.org/",
            category="News APIs",
            access_level=AccessLevel.FREE_ACCOUNT_NEEDED,
            polling_strategy=PollingStrategy.LIMITED,
            rate_limit=RateLimit(requests_per_day=100),
            description="News headlines and articles",
            api_key_required=True,
            authentication_method="API Key",
            documentation_url="https://newsapi.org/docs"
        ))
        
        self.add_source(DataSource(
            name="Reddit API",
            url="https://www.reddit.com/dev/api/",
            category="Social Media",
            access_level=AccessLevel.FREE_ACCOUNT_NEEDED,
            polling_strategy=PollingStrategy.MODERATE,
            rate_limit=RateLimit(requests_per_minute=60),
            description="Reddit posts and comments",
            api_key_required=True,
            authentication_method="OAuth",
            documentation_url="https://www.reddit.com/dev/api/"
        ))
        
        # RSS Feeds - Free (No Account) - Safe for Frequent Polling
        frequent_rss_feeds = [
            "Musings on Markets", "TraderFeed", "DivHut", "Crossing Wall Street",
            "Alpha Ideas", "Carl Futia", "Investment Talk – Susan Brunner",
            "Singapore Humble Stock", "DivGro", "My Stocks Investing",
            "SG Wealth Builder", "Ciovacco Capital Management", "Morpheus Trading Group",
            "Guerilla Stock Trading", "The Blue Collar Investor", "Terry's Tips Blog",
            "The Brown Report", "Wishing Wealth Blog", "Option Strategist Blog",
            "Reminiscences of a Stockblogger", "Lucky Idiot", "Goode Trades"
        ]
        
        for feed_name in frequent_rss_feeds:
            self.add_source(DataSource(
                name=feed_name,
                url=f"https://{feed_name.lower().replace(' ', '').replace('–', '')}.com/feed",
                category="RSS Feeds",
                access_level=AccessLevel.FREE_NO_ACCOUNT,
                polling_strategy=PollingStrategy.FREQUENT,
                rate_limit=RateLimit(requests_per_minute=4),  # Every 15 min
                description=f"Financial blog RSS feed - {feed_name}",
                tags=["rss", "blog", "financial"]
            ))
        
        # RSS Feeds - Free (No Account) - Moderate Polling
        moderate_rss_feeds = [
            "MarketWatch", "Seeking Alpha", "Investing.com", "Economic Times",
            "CNBC Market Insider", "INO.com Blog", "StockCharts.com", "Trade Brains Blog"
        ]
        
        for feed_name in moderate_rss_feeds:
            self.add_source(DataSource(
                name=feed_name,
                url=f"https://{feed_name.lower().replace(' ', '').replace('.', '')}.com/rss",
                category="RSS Feeds",
                access_level=AccessLevel.FREE_NO_ACCOUNT,
                polling_strategy=PollingStrategy.MODERATE,
                rate_limit=RateLimit(requests_per_minute=2),  # Every 30 min
                description=f"Major financial news RSS feed - {feed_name}",
                tags=["rss", "news", "major-publisher"]
            ))
        
        # Government Data Sources - 100% Free (No Account)
        government_sources = [
            ("FRED API", "https://fred.stlouisfed.org", "Federal Reserve Economic Data"),
            ("World Bank Open Data", "https://api.worldbank.org", "Global development indicators"),
            ("Data.gov", "https://data.gov", "U.S. government datasets"),
            ("OECD Statistics", "https://stats.oecd.org", "Economic and social data"),
            ("UN Data Portal", "https://data.un.org", "International statistics")
        ]
        
        for name, url, desc in government_sources:
            self.add_source(DataSource(
                name=name,
                url=url,
                category="Government Data",
                access_level=AccessLevel.FREE_NO_ACCOUNT,
                polling_strategy=PollingStrategy.MODERATE,
                rate_limit=RateLimit(custom_limit="No rate limits - public data"),
                description=desc,
                tags=["government", "official", "economic-data"]
            ))
        
        # Additional RSS Feeds - 100% Free (No Account)
        additional_rss_sources = [
            ("Reuters Business RSS", "https://www.reuters.com/business/finance/rss", "Global business news"),
            ("Benzinga RSS", "https://feeds.benzinga.com/benzinga", "Stock market news"),
            ("Seeking Alpha RSS", "https://seekingalpha.com/feed.xml", "Investment analysis"),
            ("The Motley Fool RSS", "https://www.fool.com/feeds/index.xml", "Investment advice"),
            ("ZeroHedge RSS", "https://www.zerohedge.com/fullrss2.xml", "Alternative finance news"),
            ("Financial Times RSS", "https://www.ft.com/?format=rss", "Business headlines"),
            ("Investopedia RSS", "https://feeds.feedburner.com/InvestopediaLatest", "Financial education"),
            ("Kiplinger RSS", "https://feeds.feedburner.com/kiplinger/latest", "Personal finance")
        ]
        
        for name, url, desc in additional_rss_sources:
            self.add_source(DataSource(
                name=name,
                url=url,
                category="RSS Feeds",
                access_level=AccessLevel.FREE_NO_ACCOUNT,
                polling_strategy=PollingStrategy.MODERATE,
                rate_limit=RateLimit(requests_per_minute=2),
                description=desc,
                tags=["rss", "news", "no-account"]
            ))
        
        # TradingView Widgets - 100% Free (No Account)
        self.add_source(DataSource(
            name="TradingView Widgets",
            url="https://www.tradingview.com/widget/",
            category="Market Data Widgets",
            access_level=AccessLevel.FREE_NO_ACCOUNT,
            polling_strategy=PollingStrategy.FREQUENT,
            rate_limit=RateLimit(custom_limit="No limits - embedded widgets"),
            description="Free embeddable charts and market data widgets",
            tags=["widgets", "charts", "real-time"]
        ))
        
        # Additional Free (No Account) APIs
        no_account_apis = [
            ("OpenWeatherMap Free", "https://openweathermap.org/api", "Weather data for financial centers", 60, None),
            ("Wikipedia API", "https://en.wikipedia.org/api/rest_v1", "Company and economic context", 200, None),
            ("Binance Public API", "https://api.binance.com", "Crypto market data", None, 100),
            ("Stack Overflow API", "https://api.stackexchange.com", "Financial programming Q&A", None, 30),
            ("OECD Statistics", "https://stats.oecd.org", "Economic indicators", None, None)
        ]
        
        for name, url, desc, minute_limit, second_limit in no_account_apis:
            rate_limit = RateLimit()
            if minute_limit:
                rate_limit.requests_per_minute = minute_limit
            if second_limit:
                rate_limit.custom_limit = f"{second_limit} requests/second"
                
            # Determine if account is needed
            access_level = AccessLevel.FREE_NO_ACCOUNT
            api_key_required = False
            
            # OpenWeatherMap requires API key but has free tier
            if "OpenWeatherMap" in name:
                access_level = AccessLevel.FREE_ACCOUNT_NEEDED
                api_key_required = True
            
            self.add_source(DataSource(
                name=name,
                url=url,
                category="Additional APIs",
                access_level=access_level,
                polling_strategy=PollingStrategy.MODERATE,
                rate_limit=rate_limit,
                description=desc,
                api_key_required=api_key_required,
                authentication_method="API Key" if api_key_required else None
            ))
        
        # Paid Services
        paid_services = [
            ("Polygon.io", "https://polygon.io/", "Real-time market data", "5-100 requests/sec"),
            ("NewsAPI Pro", "https://newsapi.org/pricing", "Full-text articles", "Varies by plan"),
            ("Morningstar Direct", "https://www.morningstar.com/products/direct", "Professional financial data", "Custom rate limits"),
            ("Refinitiv Datastream", "https://www.refinitiv.com/en/products/datastream-macroeconomic-analysis", "Market and economic data", "Enterprise rate limits")
        ]
        
        for name, url, desc, rate in paid_services:
            self.add_source(DataSource(
                name=name,
                url=url,
                category="Premium APIs",
                access_level=AccessLevel.PAID_ACCOUNT_NEEDED,
                polling_strategy=PollingStrategy.LIMITED,
                rate_limit=RateLimit(custom_limit=rate),
                description=desc,
                api_key_required=True,
                authentication_method="API Key / OAuth",
                tags=["premium", "professional"]
            ))
    
    def add_source(self, source: DataSource):
        """Add a data source to the registry"""
        self.sources[source.name] = source
    
    def get_sources_by_access_level(self, access_level: AccessLevel) -> List[DataSource]:
        """Get all sources by access level"""
        return [source for source in self.sources.values() if source.access_level == access_level]
    
    def get_sources_by_category(self, category: str) -> List[DataSource]:
        """Get all sources by category"""
        return [source for source in self.sources.values() if source.category == category]
    
    def get_active_sources(self) -> List[DataSource]:
        """Get all active sources"""
        return [source for source in self.sources.values() if source.status == "active"]
    
    def get_source(self, name: str) -> Optional[DataSource]:
        """Get a specific source by name"""
        return self.sources.get(name)
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(source.category for source in self.sources.values()))
    
    def get_sources_requiring_api_key(self) -> List[DataSource]:
        """Get sources that require API keys"""
        return [source for source in self.sources.values() if source.api_key_required]
    
    def get_rate_limit_info(self, source_name: str) -> Optional[RateLimit]:
        """Get rate limit information for a source"""
        source = self.get_source(source_name)
        return source.rate_limit if source else None
    
    def export_source_list(self) -> Dict:
        """Export sources as JSON-serializable dict"""
        return {
            name: {
                "name": source.name,
                "url": source.url,
                "category": source.category,
                "access_level": source.access_level.value,
                "polling_strategy": source.polling_strategy.value,
                "rate_limit": {
                    "requests_per_minute": source.rate_limit.requests_per_minute,
                    "requests_per_hour": source.rate_limit.requests_per_hour,
                    "requests_per_day": source.rate_limit.requests_per_day,
                    "concurrent_connections": source.rate_limit.concurrent_connections,
                    "custom_limit": source.rate_limit.custom_limit
                },
                "description": source.description,
                "api_key_required": source.api_key_required,
                "authentication_method": source.authentication_method,
                "status": source.status,
                "tags": source.tags
            }
            for name, source in self.sources.items()
        }