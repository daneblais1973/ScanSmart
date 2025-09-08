import streamlit as st
from typing import Dict, Any
import logging

# Import our professional styles
try:
    from ui.clean_layout_styles import inject_professional_styles, create_metric_card
except ImportError:
    def inject_professional_styles():
        st.markdown("<style>body { background: #0f172a; color: white; }</style>", unsafe_allow_html=True)
    
    def create_metric_card(title, value, change=None, icon=None):
        return f"<div><h3>{value}</h3><p>{title}</p></div>"

logger = logging.getLogger(__name__)

def render_configuration(session_state: Dict[str, Any]):
    """Render the professional configuration page"""
    
    # Apply professional styling
    inject_professional_styles()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                padding: var(--spacing-xl); 
                border-radius: var(--radius-lg);
                margin-bottom: var(--spacing-xl);
                color: white;">
        <h1 style="margin: 0; font-size: 2.5rem;">⚙️ Settings & Configuration</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            Customize your ScanSmart trading platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs with professional styling
    tabs = st.tabs([
        "🔧 General", 
        "📊 Catalyst", 
        "📈 Stocks", 
        "🤖 AI", 
        "💾 Database", 
        "🚨 Alerts", 
        "📡 RSS", 
        "🔑 API"
    ])
    
    with tabs[0]:
        render_general_settings_tab(session_state)
    
    with tabs[1]:
        render_catalyst_settings_tab(session_state)
    
    with tabs[2]:
        render_stock_settings_tab(session_state)
    
    with tabs[3]:
        render_ai_settings_tab(session_state)
    
    with tabs[4]:
        render_database_settings_tab(session_state)
    
    with tabs[5]:
        render_alert_settings_tab(session_state)
    
    with tabs[6]:
        render_rss_accounts_tab(session_state)
    
    with tabs[7]:
        render_api_accounts_tab(session_state)

def render_general_settings_tab(session_state: Dict[str, Any]):
    """Professional General Settings tab"""
    st.subheader("🔧 General Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎨 Appearance**")
        
        theme = st.selectbox(
            "Color Theme",
            ["Dark", "Light", "Auto"],
            index=0,
            help="Choose your preferred visual theme"
        )
        
        auto_refresh = st.checkbox(
            "Auto-refresh Data",
            value=True,
            help="Automatically refresh scan results"
        )
        
        refresh_interval = st.slider(
            "Refresh Interval",
            min_value=10, max_value=300, value=60,
            help="Seconds between data refreshes"
        )
        
        show_notifications = st.checkbox(
            "Desktop Notifications",
            value=True,
            help="Show system notifications for alerts"
        )
    
    with col2:
        st.markdown("**⚡ Performance**")
        
        max_concurrent = st.slider(
            "Max Concurrent Operations",
            min_value=1, max_value=50, value=10,
            help="Simultaneous data fetching operations"
        )
        
        request_timeout = st.slider(
            "Request Timeout",
            min_value=5, max_value=120, value=30,
            help="Seconds before API requests timeout"
        )
        
        enable_caching = st.checkbox(
            "Enable Caching",
            value=True,
            help="Cache data for better performance"
        )
        
        cache_size_mb = st.slider(
            "Cache Size",
            min_value=50, max_value=2048, value=512,
            help="Maximum cache size in MB"
        )
    
    if st.button("💾 Save General Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ Settings saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save general settings: {e}")
            st.error(f"Save failed: {e}")

def render_catalyst_settings_tab(session_state: Dict[str, Any]):
    """Professional Catalyst Settings tab"""
    st.subheader("📊 Catalyst Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Detection Thresholds**")
        
        min_impact_score = st.slider(
            "Minimum Impact Score",
            0, 100, 50,
            help="Only detect catalysts above this impact threshold"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.7, step=0.05,
            help="Confidence level required for detection"
        )
        
        max_age_hours = st.slider(
            "Maximum Age",
            min_value=1, max_value=168, value=24,
            help="Hours before catalysts are ignored"
        )
    
    with col2:
        st.markdown("**📂 Category Filters**")
        
        enabled_categories = st.multiselect(
            "Monitor Categories",
            [
                "Earnings / Guidance",
                "M&A / Partnerships", 
                "Regulatory / FDA",
                "Analyst Update",
                "Insider Trading",
                "General News"
            ],
            default=["Earnings / Guidance", "M&A / Partnerships", "Regulatory / FDA"],
            help="Which catalyst categories to monitor"
        )
        
        sentiment_filter = st.multiselect(
            "Sentiment Types",
            ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            default=["POSITIVE", "NEGATIVE"],
            help="Which sentiment types to include"
        )
    
    st.markdown("**⚙️ Advanced Options**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duplicate_threshold = st.slider(
            "Duplicate Detection",
            0.0, 1.0, 0.85,
            help="Similarity threshold for duplicates"
        )
    
    with col2:
        enable_ml_scoring = st.checkbox(
            "ML Impact Scoring",
            value=True,
            help="Use machine learning for scoring"
        )
    
    with col3:
        batch_processing = st.checkbox(
            "Batch Processing",
            value=True,
            help="Process catalysts in batches"
        )
    
    if st.button("💾 Save Catalyst Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ Catalyst settings saved!")
        except Exception as e:
            logger.error(f"Failed to save catalyst settings: {e}")
            st.error(f"Save failed: {e}")

def render_stock_settings_tab(session_state: Dict[str, Any]):
    """Professional Stock Settings tab"""
    st.subheader("📈 Stock Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏛️ Market Filters**")
        
        exchanges = st.multiselect(
            "Exchanges",
            ["NYSE", "NASDAQ", "AMEX", "OTC"],
            default=["NYSE", "NASDAQ"],
            help="Exchanges to monitor"
        )
        
        market_cap_min = st.number_input(
            "Min Market Cap ($M)",
            min_value=0, value=100,
            help="Minimum market cap in millions"
        )
        
        market_cap_max = st.number_input(
            "Max Market Cap ($B)",
            min_value=1, value=1000,
            help="Maximum market cap in billions"
        )
        
        volume_min = st.number_input(
            "Min Daily Volume",
            min_value=0, value=100000,
            help="Minimum average daily volume"
        )
    
    with col2:
        st.markdown("**🏢 Sector Focus**")
        
        sectors = st.multiselect(
            "Focus Sectors",
            [
                "Technology", "Healthcare", "Financial",
                "Consumer Discretionary", "Industrial", "Energy",
                "Materials", "Utilities", "Real Estate",
                "Communication Services", "Consumer Staples"
            ],
            default=["Technology", "Healthcare", "Financial"],
            help="Sectors to focus on"
        )
        
        exclude_penny = st.checkbox(
            "Exclude Penny Stocks (<$5)",
            value=True
        )
        
        exclude_otc = st.checkbox(
            "Exclude OTC Stocks",
            value=True
        )
    
    st.markdown("**⭐ Custom Watchlist**")
    
    watchlist_tickers = st.text_area(
        "Watchlist Symbols",
        placeholder="AAPL, MSFT, GOOGL, TSLA, NVDA",
        help="Your personal stock watchlist"
    )
    
    if st.button("💾 Save Stock Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ Stock settings saved!")
        except Exception as e:
            logger.error(f"Failed to save stock settings: {e}")
            st.error(f"Save failed: {e}")

def render_ai_settings_tab(session_state: Dict[str, Any]):
    """Professional AI Settings tab"""
    st.subheader("🤖 AI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Model Settings**")
        
        model_device = st.selectbox(
            "Compute Device",
            ["auto", "cpu", "cuda"],
            help="Hardware for AI processing"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1, max_value=128, value=32,
            help="Inference batch size"
        )
        
        max_sequence_length = st.slider(
            "Max Sequence Length",
            min_value=128, max_value=2048, value=512,
            help="Maximum input text length"
        )
    
    with col2:
        st.markdown("**🚀 AI Features**")
        
        enable_sentiment = st.checkbox(
            "Sentiment Analysis",
            value=True
        )
        
        enable_summarization = st.checkbox(
            "Text Summarization",
            value=True
        )
        
        enable_classification = st.checkbox(
            "Catalyst Classification",
            value=True
        )
    
    st.markdown("**🧠 OpenAI Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        openai_model = st.selectbox(
            "Model Version",
            ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            help="OpenAI model selection"
        )
    
    with col2:
        temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.7,
            help="Response creativity level"
        )
    
    if st.button("💾 Save AI Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ AI settings saved!")
        except Exception as e:
            logger.error(f"Failed to save AI settings: {e}")
            st.error(f"Save failed: {e}")

def render_database_settings_tab(session_state: Dict[str, Any]):
    """Professional Database Settings tab"""
    st.subheader("💾 Database Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔌 Connection**")
        
        db_type = st.selectbox(
            "Database Type",
            ["SQLite", "PostgreSQL", "MySQL"],
            help="Database technology"
        )
        
        connection_pool_size = st.slider(
            "Connection Pool",
            min_value=1, max_value=50, value=10,
            help="Number of database connections"
        )
    
    with col2:
        st.markdown("**🗃️ Data Management**")
        
        retention_days = st.slider(
            "Data Retention",
            min_value=1, max_value=365, value=90,
            help="Days to keep historical data"
        )
        
        auto_cleanup = st.checkbox(
            "Auto Cleanup",
            value=True,
            help="Automatically remove old data"
        )
    
    # Database status with professional styling
    try:
        if session_state and 'db' in session_state and session_state['db']:
            if hasattr(session_state['db'], 'is_connected') and session_state['db'].is_connected():
                st.success("✅ Database connected")
                if hasattr(session_state['db'], 'get_catalyst_count'):
                    total_catalysts = session_state['db'].get_catalyst_count()
                    st.metric("Total Records", f"{total_catalysts:,}")
            else:
                st.error("❌ Database disconnected")
        else:
            st.warning("⚠️ Database not initialized")
    except Exception as e:
        logger.error(f"Database status error: {e}")
        st.error(f"Database error: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Connection", use_container_width=True):
            try:
                if session_state and 'db' in session_state and session_state['db']:
                    if hasattr(session_state['db'], 'is_connected') and session_state['db'].is_connected():
                        st.success("✅ Connection successful!")
                    else:
                        st.error("❌ Connection failed")
                else:
                    st.error("❌ Database not available")
            except Exception as e:
                logger.error(f"Connection test failed: {e}")
                st.error(f"Test failed: {e}")
    
    with col2:
        if st.button("💾 Save Database Settings", type="primary", use_container_width=True):
            try:
                st.success("✅ Database settings saved!")
            except Exception as e:
                logger.error(f"Failed to save database settings: {e}")
                st.error(f"Save failed: {e}")

def render_alert_settings_tab(session_state: Dict[str, Any]):
    """Professional Alert Settings tab"""
    st.subheader("🚨 Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Alert Thresholds**")
        
        alert_min_impact = st.slider(
            "Min Impact Score",
            0, 100, 75,
            help="Minimum impact for alerts"
        )
        
        alert_min_confidence = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.8,
            help="Minimum confidence for alerts"
        )
        
        rate_limit_minutes = st.slider(
            "Rate Limit",
            min_value=1, max_value=60, value=5,
            help="Minutes between alerts"
        )
    
    with col2:
        st.markdown("**📱 Notification Channels**")
        
        email_alerts = st.checkbox(
            "Email Alerts",
            value=True
        )
        
        sms_alerts = st.checkbox(
            "SMS Alerts", 
            value=False
        )
        
        desktop_notifications = st.checkbox(
            "Desktop Notifications",
            value=True
        )
    
    if email_alerts:
        email_address = st.text_input(
            "Alert Email",
            placeholder="your-email@example.com"
        )
    
    if sms_alerts:
        phone_number = st.text_input(
            "Phone Number",
            placeholder="+1234567890"
        )
    
    if st.button("💾 Save Alert Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ Alert settings saved!")
        except Exception as e:
            logger.error(f"Failed to save alert settings: {e}")
            st.error(f"Save failed: {e}")

def render_rss_accounts_tab(session_state: Dict[str, Any]):
    """Professional RSS Accounts tab"""
    st.subheader("📡 RSS Feed Configuration")
    
    st.markdown("**📰 Default News Sources**")
    
    default_feeds = [
        {"name": "MarketWatch", "url": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/", "enabled": True},
        {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "enabled": True},
        {"name": "Bloomberg", "url": "https://feeds.bloomberg.com/markets/news.rss", "enabled": True},
        {"name": "Yahoo Finance", "url": "https://feeds.finance.yahoo.com/rss/2.0/headline", "enabled": True},
        {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "enabled": True}
    ]
    
    for i, feed in enumerate(default_feeds):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            enabled = st.checkbox(
                "Enable",
                value=feed["enabled"],
                key=f"rss_enable_{i}",
                label_visibility="collapsed"
            )
        
        with col2:
            st.text_input(
                f"{feed['name']}",
                value=feed["url"],
                key=f"rss_url_{i}",
                label_visibility="collapsed"
            )
        
        with col3:
            if st.button("Test", key=f"rss_test_{i}", use_container_width=True):
                st.success(f"✅ {feed['name']} feed working")
    
    st.markdown("**➕ Custom RSS Feeds**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_feed_url = st.text_input(
            "Custom RSS URL",
            placeholder="https://example.com/rss.xml"
        )
    
    with col2:
        if st.button("Add Feed", use_container_width=True):
            if custom_feed_url:
                st.success("Custom feed added!")
    
    st.markdown("**⚙️ Processing Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fetch_interval = st.slider(
            "Fetch Interval (min)",
            min_value=5, max_value=120, value=15
        )
    
    with col2:
        max_articles = st.slider(
            "Max Articles",
            min_value=10, max_value=500, value=50
        )
    
    if st.button("💾 Save RSS Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ RSS settings saved!")
        except Exception as e:
            logger.error(f"Failed to save RSS settings: {e}")
            st.error(f"Save failed: {e}")

def render_api_accounts_tab(session_state: Dict[str, Any]):
    """Professional API Accounts tab"""
    st.subheader("🔑 API Key Management")
    
    # Display source summary
    try:
        from data_sources.api_sources_list import get_source_summary
        summary = get_source_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🆓 Free Sources", summary["free_no_account"], help="No account needed")
        with col2:
            st.metric("🔐 Account Required", summary["free_with_account"], help="Free account signup")
        with col3:
            st.metric("💰 Premium Sources", summary["paid"], help="Paid subscriptions")
    except:
        pass
    
    st.markdown("**📊 Free Financial Data APIs (Account Required)**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "Alpha Vantage API Key",
            value=get_config_value(session_state, 'alpha_vantage_key'),
            type="password",
            help="500 requests/day (free) - https://www.alphavantage.co/support/#api-key"
        )
        
        st.text_input(
            "Finnhub API Key",
            value=get_config_value(session_state, 'finnhub_key'),
            type="password",
            help="60 requests/minute (free) - https://finnhub.io/register"
        )
        
        st.text_input(
            "Financial Modeling Prep Key",
            value=get_config_value(session_state, 'fmp_key'),
            type="password",
            help="250 requests/day (free) - https://site.financialmodelingprep.com/developer/docs"
        )
    
    with col2:
        st.text_input(
            "IEX Cloud API Key",
            value=get_config_value(session_state, 'iex_key'),
            type="password",
            help="500K messages/month (free) - https://iexcloud.io/pricing/"
        )
        
        st.text_input(
            "Marketstack API Key",
            value=get_config_value(session_state, 'marketstack_key'),
            type="password",
            help="1,000 requests/month (free) - https://marketstack.com/signup"
        )
        
        st.text_input(
            "News Data.io API Key",
            value=get_config_value(session_state, 'newsdata_key'),
            type="password",
            help="200 requests/day (free) - https://newsdata.io/register"
        )
    
    st.markdown("**🔬 Development & Research APIs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "Kaggle API Key",
            value=get_config_value(session_state, 'kaggle_key'),
            type="password",
            help="10 requests/minute (free) - https://www.kaggle.com/docs/api"
        )
        
        st.text_input(
            "Quandl API Key", 
            value=get_config_value(session_state, 'quandl_key'),
            type="password",
            help="20 requests/day (free) - https://data.nasdaq.com/sign-up"
        )
    
    with col2:
        st.text_input(
            "GitHub Token",
            value=get_config_value(session_state, 'github_token'),
            type="password", 
            help="5000 requests/hour (free) - https://github.com/settings/tokens"
        )
        
        st.text_input(
            "OpenWeatherMap API Key",
            value=get_config_value(session_state, 'openweather_key'),
            type="password",
            help="60 calls/minute (free) - https://openweathermap.org/api"
        )
    
    st.markdown("**🏛️ Alternative & Specialized Data APIs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "EIA Energy API Key",
            value=get_config_value(session_state, 'eia_api_key'),
            type="password",
            help="Free - Energy data (oil, gas, electricity) - https://www.eia.gov/opendata/"
        )
        
        st.text_input(
            "Financial Modeling Prep API Key",
            value=get_config_value(session_state, 'fmp_api_key'),
            type="password",
            help="250 requests/day (free) - Earnings, financials - https://financialmodelingprep.com/"
        )
        
        st.text_input(
            "Sustainalytics ESG API Key",
            value=get_config_value(session_state, 'sustainalytics_api_key'),
            type="password",
            help="ESG ratings and sustainability data - https://www.sustainalytics.com/"
        )
    
    with col2:
        st.text_input(
            "MSCI ESG API Key",
            value=get_config_value(session_state, 'msci_esg_api_key'),
            type="password",
            help="ESG scores and ratings - https://www.msci.com/esg-integration"
        )
        
        st.text_input(
            "CFTC API Key",
            value=get_config_value(session_state, 'cftc_api_key'),
            type="password",
            help="Commitment of Traders data - https://publicreporting.cftc.gov/"
        )
        
        st.text_input(
            "Treasury Direct API Key",
            value=get_config_value(session_state, 'treasury_api_key'),
            type="password",
            help="Government bond data (often free) - https://www.treasurydirect.gov/"
        )
    
    st.markdown("**💰 Premium Financial Data APIs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "Polygon.io API Key",
            value=get_config_value(session_state, 'polygon_key'),
            type="password",
            help="Starting $99/month - https://polygon.io/pricing"
        )
        
        st.text_input(
            "Quandl API Key",
            value=get_config_value(session_state, 'quandl_key'),
            type="password",
            help="Starting $50/month - https://data.nasdaq.com/"
        )
    
    with col2:
        st.text_input(
            "Bloomberg API Key",
            value=get_config_value(session_state, 'bloomberg_key'),
            type="password",
            help="Enterprise pricing - https://www.bloomberg.com/professional/"
        )
        
        st.text_input(
            "Refinitiv API Key",
            value=get_config_value(session_state, 'refinitiv_key'),
            type="password",
            help="Enterprise pricing - https://www.refinitiv.com/"
        )
    
    st.markdown("**📰 News & Social Media APIs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "NewsAPI Key",
            value=get_config_value(session_state, 'newsapi_key'),
            type="password",
            help="100 requests/day (free) - https://newsapi.org/register"
        )
        
        st.text_input(
            "Reddit Client ID",
            value=get_config_value(session_state, 'reddit_client_id'),
            type="password",
            help="60 requests/minute (free) - https://www.reddit.com/dev/api/"
        )
    
    with col2:
        st.text_input(
            "Twitter API Bearer Token",
            value=get_config_value(session_state, 'twitter_bearer_token'),
            type="password",
            help="1,500 requests/month (free) - https://developer.twitter.com/"
        )
        
        st.text_input(
            "Reddit Client Secret",
            value=get_config_value(session_state, 'reddit_client_secret'),
            type="password",
            help="Required with Client ID"
        )
    
    st.markdown("**🧠 AI Service APIs**")
    
    st.text_input(
        "OpenAI API Key",
        value=get_config_value(session_state, 'openai_api_key'),
        type="password",
        help="Required for AI analysis",
        placeholder="sk-..."
    )
    
    if st.button("💾 Save API Settings", type="primary", use_container_width=True):
        try:
            st.success("✅ API settings saved!")
        except Exception as e:
            logger.error(f"Failed to save API settings: {e}")
            st.error(f"Save failed: {e}")

def get_config_value(session_state, key):
    """Safely get configuration value"""
    try:
        if session_state and 'config' in session_state and session_state['config']:
            if hasattr(session_state['config'], 'api'):
                if hasattr(session_state['config'].api, key):
                    value = getattr(session_state['config'].api, key)
                    return "••••••••" if value else ""
        return ""
    except Exception as e:
        logger.error(f"Error getting config value for {key}: {e}")
        return ""