# Overview

This is a comprehensive financial catalyst scanning and analysis system built with Python. The application monitors multiple data sources (news APIs, social media, regulatory filings) to detect market-moving events for stock symbols, performs NLP analysis on the collected data to identify catalysts and sentiment, and provides real-time alerting capabilities through various notification channels.

The system is designed as a multi-component architecture with both microservices and monolithic deployment options, featuring a Streamlit frontend for user interaction and various backend services for data processing.

# Recent Changes

## 2025-01-06
- **Fixed Syntax Errors**: Resolved multiple syntax errors in app.py including broken indentation and duplicate function definitions
- **Cleaned Up Sidebar Code**: Removed duplicate sidebar rendering code from app.py and ensured proper integration with ui/sliding_sidebar.py module
- **Application Structure**: Fixed function definition conflicts and ensured proper code organization between main app and UI components

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Application**: Single-page dashboard application (`app.py`) providing user interface for catalyst scanning, configuration, and data visualization
- **UI Components**: Modular UI components organized in `ui/` directory for dashboard, catalyst viewer, and configuration pages
- **Data Visualization**: Uses Plotly for interactive charts and Pandas for data manipulation

## Backend Architecture
- **Monolithic Core**: Main application logic in `app.py` with modular components for database, caching, NLP processing, and alerting
- **Data Fetchers**: Plugin-based data fetching system supporting multiple sources (NewsAPI, Twitter, Reddit, RSS, financial APIs, regulatory sources)
- **NLP Processing**: Advanced natural language processing pipeline using transformer models (FinBERT, BART) and spaCy for financial catalyst detection and sentiment analysis
- **Batch Processing**: Scheduled background processing system for automated data collection and analysis
- **Alert System**: Multi-channel notification system supporting email, SMS, console output, and webhooks

## Data Storage Solutions
- **Primary Database**: SQLAlchemy-based ORM with support for PostgreSQL and SQLite fallback
- **Caching Layer**: Dual caching strategy with Redis primary and in-memory fallback for performance optimization
- **Configuration Storage**: File-based JSON configuration with environment variable overrides

## Authentication and Authorization
- **API Key Management**: Support for multiple API key configurations for external data sources
- **Session Management**: Streamlit session state for user preferences and cached data
- **Service Authentication**: Basic API key authentication for internal service communication

## Data Processing Pipeline
- **Multi-Source Ingestion**: Concurrent data fetching from news APIs, social media platforms, and regulatory sources
- **NLP Analysis**: Financial-specific sentiment analysis and catalyst categorization using pre-trained financial language models
- **Deduplication**: Content similarity detection to prevent duplicate catalyst alerts
- **Impact Scoring**: Proprietary scoring algorithm considering sentiment, source credibility, and market timing

# External Dependencies

## Third-Party APIs
- **NewsAPI.org**: Primary news data source requiring API key
- **Twitter API v2**: Social media sentiment data via Bearer token authentication
- **Reddit API**: Community discussion analysis through OAuth client credentials
- **Financial APIs**: Financial Modeling Prep, Alpha Vantage, and Polygon.io for market data
- **Regulatory APIs**: SEC EDGAR and FDA APIs for regulatory filing monitoring
- **Web Scraping Services**: ScrapingBee, ScrapingDog, and BrightData for alternative data sources

## Core Python Libraries
- **FastAPI/Uvicorn**: High-performance async web framework for potential microservice deployment
- **Streamlit**: Frontend dashboard framework
- **SQLAlchemy**: Database ORM with PostgreSQL driver (psycopg2-binary)
- **Redis**: Caching and session storage
- **aiohttp/httpx**: Asynchronous HTTP client libraries for API communication

## Machine Learning Stack
- **Transformers/PyTorch**: Pre-trained financial language models (FinBERT, BART)
- **spaCy**: Named entity recognition and text processing
- **scikit-learn**: TF-IDF vectorization and similarity analysis
- **NLTK**: Natural language processing utilities

## Data Processing Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualization
- **feedparser**: RSS feed processing

## Infrastructure Libraries
- **schedule**: Task scheduling for batch processing
- **celery**: Distributed task queue (optional)
- **tenacity**: Retry logic and error handling
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation and settings management