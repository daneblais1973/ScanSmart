"""
Categorized list of financial data sources by access level
Helps users understand what requires accounts/APIs vs what's completely free
"""
from typing import Dict, List

# 100% FREE SOURCES (No Account/API Key Required)
FREE_NO_ACCOUNT_SOURCES = {
    "RSS Feeds": [
        "Yahoo Finance RSS",
        "MarketWatch RSS", 
        "CNBC RSS",
        "CNN Money RSS",
        "Bloomberg RSS",
        "Dow Jones RSS",
        "Reuters Business RSS",
        "Benzinga RSS",
        "Seeking Alpha RSS", 
        "The Motley Fool RSS",
        "ZeroHedge RSS",
        "Financial Times RSS",
        "Investopedia RSS",
        "Kiplinger RSS"
    ],
    "Government/Economic Data": [
        "FRED (Federal Reserve Economic Data)",
        "World Bank Open Data",
        "Data.gov",
        "OECD Statistics", 
        "UN Data Portal",
        "US Treasury Direct",
        "IMF Data",
        "Eurostat"
    ],
    "Cryptocurrency": [
        "CoinGecko API",
        "CoinDesk API",
        "Binance Public API"
    ],
    "Market Widgets": [
        "TradingView Widgets",
        "Yahoo Finance CSV Export"
    ],
    "Weather Data": [
        "OpenWeatherMap (Free Tier - 60 calls/min)"
    ],
    "Knowledge/Research": [
        "Wikipedia API (200 requests/sec)",
        "USPTO Patent Database (No limits)"
    ],
    "Central Bank Data": [
        "European Central Bank (ECB) - No limits",
        "Bank of Japan Economic Data",
        "Bank of England Statistics"
    ],
    "Energy & Commodities": [
        "CBOE VIX and Options (Free)",
        "US Treasury Direct Bond Data"
    ],
    "ESG & Sustainability": [
        "Public ESG Databases",
        "CDP Climate Data",
        "UN Global Compact Database"
    ],
    "Web Scraping": [
        "Yahoo Finance (yfinance library)",
        "Investing.com Economic Calendar",
        "MarketWatch data"
    ]
}

# FREE WITH ACCOUNT (Easy 2-minute signup)
FREE_WITH_ACCOUNT_SOURCES = {
    "Financial APIs": [
        {
            "name": "Alpha Vantage",
            "limit": "500 requests/day",
            "signup_url": "https://www.alphavantage.co/support/#api-key",
            "data": "Stocks, forex, crypto, technical indicators",
            "api_key_field": "alpha_vantage_key"
        },
        {
            "name": "Finnhub",
            "limit": "60 requests/minute",
            "signup_url": "https://finnhub.io/register",
            "data": "Real-time stocks, company fundamentals",
            "api_key_field": "finnhub_key"
        },
        {
            "name": "Financial Modeling Prep",
            "limit": "250 requests/day",
            "signup_url": "https://site.financialmodelingprep.com/developer/docs",
            "data": "Financial statements, ratios, prices",
            "api_key_field": "fmp_key"
        },
        {
            "name": "IEX Cloud",
            "limit": "500,000 messages/month",
            "signup_url": "https://iexcloud.io/pricing/",
            "data": "US stock market data",
            "api_key_field": "iex_key"
        },
        {
            "name": "Marketstack",
            "limit": "1,000 requests/month",
            "signup_url": "https://marketstack.com/signup",
            "data": "End-of-day stock data",
            "api_key_field": "marketstack_key"
        },
        {
            "name": "Quandl Free Tier",
            "limit": "20 requests/day",
            "signup_url": "https://data.nasdaq.com/sign-up",
            "data": "Economic and financial datasets",
            "api_key_field": "quandl_key"
        }
    ],
    "News APIs": [
        {
            "name": "NewsAPI",
            "limit": "100 requests/day",
            "signup_url": "https://newsapi.org/register",
            "data": "News headlines and articles",
            "api_key_field": "newsapi_key"
        },
        {
            "name": "News Data.io",
            "limit": "200 requests/day",
            "signup_url": "https://newsdata.io/register",
            "data": "Global news aggregation",
            "api_key_field": "newsdata_key"
        }
    ],
    "Alternative Data APIs": [
        {
            "name": "EIA Energy Data",
            "limit": "5000 requests/hour",
            "signup_url": "https://www.eia.gov/opendata/",
            "data": "Oil, gas, electricity, renewable energy data",
            "api_key_field": "eia_api_key"
        },
        {
            "name": "Financial Modeling Prep",
            "limit": "250 requests/day",
            "signup_url": "https://financialmodelingprep.com/developer/docs",
            "data": "Earnings transcripts, financial statements",
            "api_key_field": "fmp_api_key"
        },
        {
            "name": "CBOE Market Data",
            "limit": "No limit (free tier)",
            "signup_url": "https://www.cboe.com/market_data/",
            "data": "VIX, options, volatility indices (free)",
            "api_key_field": "none"
        },
        {
            "name": "USPTO Patents",
            "limit": "No limit",
            "signup_url": "https://developer.uspto.gov/",
            "data": "Patent filings and innovation data (free)",
            "api_key_field": "none"
        }
    ],
    "ESG & Sustainability APIs": [
        {
            "name": "Sustainalytics ESG",
            "limit": "Varies by plan",
            "signup_url": "https://www.sustainalytics.com/",
            "data": "ESG risk ratings and sustainability metrics",
            "api_key_field": "sustainalytics_api_key"
        },
        {
            "name": "MSCI ESG Ratings",
            "limit": "Varies by plan",
            "signup_url": "https://www.msci.com/esg-integration",
            "data": "ESG scores and ratings",
            "api_key_field": "msci_esg_api_key"
        }
    ],
    "Development/Research APIs": [
        {
            "name": "Kaggle API",
            "limit": "10 requests/minute",
            "signup_url": "https://www.kaggle.com/docs/api",
            "data": "Financial datasets and competitions",
            "api_key_field": "kaggle_key"
        },
        {
            "name": "GitHub API",
            "limit": "60/hour unauth, 5000/hour auth",
            "signup_url": "https://github.com/settings/tokens",
            "data": "Fintech repositories and development trends",
            "api_key_field": "github_token"
        },
        {
            "name": "Stack Overflow API",
            "limit": "300 requests/day, 30/sec",
            "signup_url": "https://api.stackexchange.com/",
            "data": "Financial programming questions (no key required)",
            "api_key_field": "none"
        }
    ],
    "Social Media": [
        {
            "name": "Reddit API",
            "limit": "60 requests/minute",
            "signup_url": "https://www.reddit.com/dev/api/",
            "data": "Reddit posts and sentiment",
            "api_key_field": "reddit_client_id"
        },
        {
            "name": "Twitter API v2",
            "limit": "1,500 requests/month",
            "signup_url": "https://developer.twitter.com/",
            "data": "Tweets and social sentiment",
            "api_key_field": "twitter_bearer_token"
        }
    ]
}

# PAID/PREMIUM SOURCES
PAID_SOURCES = {
    "Premium Financial APIs": [
        {
            "name": "Polygon.io",
            "pricing": "Starting $99/month",
            "data": "Real-time market data, unlimited requests",
            "url": "https://polygon.io/pricing",
            "api_key_field": "polygon_key"
        },
        {
            "name": "Quandl (NASDAQ Data Link)",
            "pricing": "Starting $50/month", 
            "data": "Premium financial datasets",
            "url": "https://data.nasdaq.com/",
            "api_key_field": "quandl_key"
        },
        {
            "name": "Bloomberg API",
            "pricing": "Enterprise pricing",
            "data": "Professional Bloomberg Terminal data",
            "url": "https://www.bloomberg.com/professional/support/api-library/",
            "api_key_field": "bloomberg_key"
        },
        {
            "name": "Refinitiv (Reuters) API", 
            "pricing": "Enterprise pricing",
            "data": "Professional market data",
            "url": "https://www.refinitiv.com/",
            "api_key_field": "refinitiv_key"
        }
    ],
    "Cloud Data Platforms": [
        {
            "name": "Google BigQuery",
            "pricing": "Free tier + usage pricing",
            "data": "Massive public datasets and custom queries",
            "url": "https://cloud.google.com/bigquery/pricing",
            "api_key_field": "google_cloud_key"
        },
        {
            "name": "Snowflake",
            "pricing": "Free trial + usage pricing",
            "data": "Cloud data warehouse",
            "url": "https://www.snowflake.com/pricing/",
            "api_key_field": "snowflake_key"
        },
        {
            "name": "AWS Redshift",
            "pricing": "Usage-based pricing",
            "data": "Amazon data warehouse",
            "url": "https://aws.amazon.com/redshift/pricing/",
            "api_key_field": "aws_key"
        }
    ],
    "Premium APIs": [
        {
            "name": "Google Cloud APIs",
            "pricing": "Pay per use",
            "data": "Maps, NLP, Vision, Translation",
            "url": "https://cloud.google.com/pricing",
            "api_key_field": "google_cloud_key"
        },
        {
            "name": "Diffbot",
            "pricing": "Starting $299/month",
            "data": "Structured web data extraction",
            "url": "https://www.diffbot.com/pricing/",
            "api_key_field": "diffbot_key"
        }
    ],
    "Premium News": [
        {
            "name": "Wall Street Journal API",
            "pricing": "Contact for pricing",
            "data": "WSJ articles and analysis",
            "url": "https://developer.wsj.com/",
            "api_key_field": "wsj_key"
        }
    ]
}

def get_source_summary():
    """Get summary of all source categories"""
    free_no_account_total = sum(len(sources) for sources in FREE_NO_ACCOUNT_SOURCES.values())
    free_with_account_total = sum(len(sources) for sources in FREE_WITH_ACCOUNT_SOURCES.values())
    paid_total = sum(len(sources) for sources in PAID_SOURCES.values())
    
    return {
        "free_no_account": free_no_account_total,
        "free_with_account": free_with_account_total, 
        "paid": paid_total,
        "total": free_no_account_total + free_with_account_total + paid_total
    }

def get_all_api_key_fields():
    """Get all API key field names for settings form"""
    fields = []
    
    # From free with account sources
    for category in FREE_WITH_ACCOUNT_SOURCES.values():
        for source in category:
            if isinstance(source, dict) and "api_key_field" in source:
                fields.append(source["api_key_field"])
    
    # From paid sources
    for category in PAID_SOURCES.values():
        for source in category:
            if isinstance(source, dict) and "api_key_field" in source:
                fields.append(source["api_key_field"])
    
    return list(set(fields))  # Remove duplicates