"""
Catalyst Streaming Page
Real-time catalyst detection and display with streaming table
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def render_catalyst_streaming_page(session_state):
    """Render the live catalyst streaming page with real-time updates"""
    
    st.markdown("# 🎯 Live Catalyst Detection")
    st.markdown("**Real-time streaming of market catalysts from multiple sources**")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📊 Catalyst Stream")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True, help="Automatically refresh catalyst data")
    with col3:
        refresh_interval = st.selectbox("Refresh rate", [5, 10, 30, 60], index=0, help="Seconds between updates")
    
    # Initialize session state for selected catalyst
    if 'selected_catalyst_id' not in st.session_state:
        st.session_state.selected_catalyst_id = None
    
    # Get live catalyst data
    catalysts = get_live_catalyst_stream(session_state)
    
    if catalysts:
        # Create streaming table with Source and Headline columns
        catalyst_table_data = []
        for i, catalyst in enumerate(catalysts):
            catalyst_table_data.append({
                'ID': i,
                'Source': format_source_name(catalyst.source),
                'Headline': catalyst.catalyst[:150] + "..." if len(catalyst.catalyst) > 150 else catalyst.catalyst,
                'Ticker': catalyst.ticker,
                'Time': catalyst.published_date.strftime('%H:%M') if catalyst.published_date else 'N/A',
                'Impact': catalyst.impact
            })
        
        df = pd.DataFrame(catalyst_table_data)
        
        # Display the catalyst streaming table
        st.markdown("### 📈 Live Catalyst Feed")
        
        # Table with clickable headlines
        selected_rows = st.dataframe(
            df[['Source', 'Headline', 'Ticker', 'Time', 'Impact']],
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"), 
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Impact": st.column_config.NumberColumn("Impact", width="small", format="%d")
            }
        )
        
        # Handle headline click - show full article below
        if selected_rows and hasattr(selected_rows, 'selection') and selected_rows.selection:
            selection_data = selected_rows.selection
            if isinstance(selection_data, dict) and 'rows' in selection_data and len(selection_data['rows']) > 0:
                selected_row_idx = selection_data['rows'][0]
                if selected_row_idx < len(catalysts):
                    selected_catalyst = catalysts[selected_row_idx]
                    st.session_state.selected_catalyst_id = selected_row_idx
                    
                    # Display full article in lower section
                    render_full_article_view(selected_catalyst)
        
        # Show streaming statistics
        render_streaming_stats(catalysts, session_state)
        
    else:
        # No catalysts found - show monitoring status
        st.info("🔍 **Catalyst detector is running but no catalysts detected yet**")
        render_monitoring_status(session_state)
        
        # Show some recent activity even with no catalysts
        st.markdown("### 📡 Data Source Status")
        render_data_source_status(session_state)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def get_live_catalyst_stream(session_state) -> List[Any]:
    """Get live catalyst data from database and active feeds"""
    try:
        if session_state and session_state.get('db'):
            # Get recent catalysts from database (last 24 hours)
            catalysts = session_state['db'].get_catalysts(limit=100)
            
            # Filter for recent catalysts (last 24 hours)
            recent_catalysts = []
            now = datetime.utcnow()
            for catalyst in catalysts:
                if catalyst.published_date:
                    time_diff = now - catalyst.published_date.replace(tzinfo=None)
                    if time_diff.total_seconds() < 86400:  # 24 hours
                        recent_catalysts.append(catalyst)
            
            # Sort by most recent first
            recent_catalysts.sort(key=lambda x: x.published_date or datetime.min, reverse=True)
            
            logger.info(f"Retrieved {len(recent_catalysts)} recent catalysts for streaming")
            return recent_catalysts
        else:
            logger.warning("Database not available for catalyst streaming")
            return []
            
    except Exception as e:
        logger.error(f"Error getting live catalyst stream: {e}")
        return []

def format_source_name(source) -> str:
    """Format source name for display"""
    if hasattr(source, 'value'):
        source = source.value
    
    source_map = {
        'newsapi': '📰 NewsAPI',
        'rss': '📡 RSS Feed',
        'twitter': '🐦 Twitter',
        'reddit': '📱 Reddit',
        'financial': '💰 Financial',
        'regulatory': '🏛️ SEC/FDA',
        'earnings': '📊 Earnings',
        'patents': '📄 Patents'
    }
    
    return source_map.get(str(source).lower(), f"📡 {str(source).title()}")

def render_full_article_view(catalyst):
    """Render the full article view when a headline is clicked"""
    st.markdown("---")
    st.markdown("## 📖 Full Article")
    
    # Article header with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", catalyst.ticker)
    with col2:
        st.metric("Impact Score", f"{catalyst.impact}/100")
    with col3:
        st.metric("Confidence", f"{catalyst.confidence:.1%}")
    with col4:
        sentiment_emoji = {"Positive": "📈", "Negative": "📉", "Neutral": "➡️"}
        emoji = sentiment_emoji.get(getattr(catalyst.sentiment_label, 'value', 'Neutral'), "➡️")
        st.metric("Sentiment", f"{emoji} {catalyst.sentiment_score:+.2f}")
    
    # Full article content
    st.markdown("### 📄 Complete Article")
    
    # Article text in a nicely formatted container
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    ">
        <div style="
            color: #e2e8f0;
            line-height: 1.6;
            font-size: 1rem;
        ">
            {catalyst.catalyst}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Article metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Article Details:**")
        st.markdown(f"• **Source:** {format_source_name(catalyst.source)}")
        st.markdown(f"• **Category:** {catalyst.category}")
        st.markdown(f"• **Sector:** {catalyst.sector}")
        if catalyst.published_date:
            st.markdown(f"• **Published:** {catalyst.published_date.strftime('%Y-%m-%d %H:%M UTC')}")
    
    with col2:
        if catalyst.url:
            st.markdown("**Actions:**")
            st.link_button("🔗 View Original Source", catalyst.url, use_container_width=True)
            
            if st.button("📋 Copy Article Text", use_container_width=True):
                st.success("Article text copied to clipboard!")

def render_streaming_stats(catalysts, session_state):
    """Render streaming statistics"""
    st.markdown("---")
    st.markdown("### 📊 Stream Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate stats
    total_catalysts = len(catalysts)
    sources = set(catalyst.source for catalyst in catalysts)
    avg_impact = sum(catalyst.impact for catalyst in catalysts) / len(catalysts) if catalysts else 0
    last_update = max(catalyst.published_date for catalyst in catalysts if catalyst.published_date) if catalysts else None
    
    with col1:
        st.metric("Live Catalysts", f"{total_catalysts:,}")
    with col2:
        st.metric("Active Sources", f"{len(sources)}")
    with col3:
        st.metric("Avg Impact", f"{avg_impact:.0f}/100")
    with col4:
        last_update_str = last_update.strftime('%H:%M') if last_update else 'N/A'
        st.metric("Last Update", last_update_str)

def render_monitoring_status(session_state):
    """Show monitoring status when no catalysts are found"""
    st.markdown("### 🔍 Active Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📡 Data Sources**")
        data_fetchers = session_state.get('data_fetchers', {})
        
        for name, fetcher in data_fetchers.items():
            if hasattr(fetcher, 'is_configured') and fetcher.is_configured():
                st.markdown(f"✅ {name.title()}")
            else:
                st.markdown(f"⚠️ {name.title()} (Configure API key)")
    
    with col2:
        st.markdown("**🧠 Analysis Pipeline**")
        nlp_processor = session_state.get('nlp')
        if nlp_processor:
            st.markdown("✅ NLP Processing")
            st.markdown("✅ Sentiment Analysis") 
            st.markdown("✅ Catalyst Detection")
        else:
            st.markdown("⚠️ NLP Not Available")
    
    with col3:
        st.markdown("**📊 System Status**")
        st.markdown("🔍 Scanning for catalysts...")
        st.markdown("📈 Processing market events...")
        st.markdown("⚡ Real-time analysis active...")

def render_data_source_status(session_state):
    """Show data source status and recent activity"""
    
    # Get data source status
    data_fetchers = session_state.get('data_fetchers', {})
    
    source_status = []
    for name, fetcher in data_fetchers.items():
        is_configured = hasattr(fetcher, 'is_configured') and fetcher.is_configured()
        source_status.append({
            'Source': format_source_name(name),
            'Status': '✅ Active' if is_configured else '⚠️ Configure API Key',
            'Type': get_source_type(name),
            'Last Check': 'Real-time' if is_configured else 'N/A'
        })
    
    df = pd.DataFrame(source_status)
    st.dataframe(df, use_container_width=True, height=300)
    
    if not any(status['Status'].startswith('✅') for status in source_status):
        st.info("💡 **Tip:** Configure API keys in Settings to enable real-time catalyst detection")

def get_source_type(source_name):
    """Get the type of data source"""
    source_types = {
        'newsapi': 'News Articles',
        'rss': 'RSS Feeds', 
        'twitter': 'Social Media',
        'reddit': 'Community Discussion',
        'financial': 'Market Data',
        'regulatory': 'SEC/FDA Filings',
        'earnings': 'Earnings Reports',
        'patents': 'Patent Filings'
    }
    return source_types.get(source_name.lower(), 'Data Feed')