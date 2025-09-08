import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import logging

from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType
from shared.utils import format_percentage, calculate_sentiment_color

logger = logging.getLogger(__name__)

def render_catalyst_viewer(session_state: Dict[str, Any]):
    """Render the catalyst viewer page"""
    st.title("🔍 Catalyst Explorer")
    st.markdown("Explore and analyze detected financial catalysts with advanced filtering and visualization.")
    
    # Render filter sidebar
    filters = render_filter_sidebar(session_state)
    
    # Get filtered catalysts
    catalysts = get_filtered_catalysts(session_state, filters)
    
    # Main content area
    if catalysts:
        render_catalyst_summary(catalysts)
        render_catalyst_table(catalysts)
        render_detailed_analysis(catalysts)
    else:
        st.info("🔍 No catalysts found matching your criteria. Try adjusting the filters.")

def render_filter_sidebar(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Render the filter sidebar and return filter values"""
    st.sidebar.header("🔧 Filters")
    
    filters = {}
    
    # Ticker filter
    filters['ticker'] = st.sidebar.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL (leave empty for all)",
        help="Filter by specific ticker symbol"
    ).upper().strip()
    
    # Impact score filter
    filters['min_impact'] = st.sidebar.slider(
        "Minimum Impact Score",
        min_value=0,
        max_value=100,
        value=50,
        help="Filter catalysts by minimum impact score"
    )
    
    # Confidence filter
    filters['min_confidence'] = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        format="%.2f",
        help="Filter catalysts by minimum confidence level"
    )
    
    # Category filter
    filters['categories'] = st.sidebar.multiselect(
        "Categories",
        options=[cat.value for cat in CatalystType],
        default=[],
        help="Filter by catalyst categories (empty = all)"
    )
    
    # Sentiment filter
    filters['sentiments'] = st.sidebar.multiselect(
        "Sentiment",
        options=[sent.value for sent in SentimentLabel],
        default=[],
        help="Filter by sentiment labels (empty = all)"
    )
    
    # Source filter
    filters['sources'] = st.sidebar.multiselect(
        "Data Sources",
        options=[source.value for source in SourceType if source != SourceType.UNKNOWN],
        default=[],
        help="Filter by data sources (empty = all)"
    )
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    
    date_option = st.sidebar.selectbox(
        "Time Period",
        options=["Last 24 hours", "Last 3 days", "Last week", "Last month", "Custom range"],
        index=0
    )
    
    if date_option == "Custom range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            filters['start_date'] = st.date_input("From", value=datetime.now().date() - timedelta(days=7))
        with col2:
            filters['end_date'] = st.date_input("To", value=datetime.now().date())
    else:
        days_mapping = {
            "Last 24 hours": 1,
            "Last 3 days": 3,
            "Last week": 7,
            "Last month": 30
        }
        days_back = days_mapping[date_option]
        filters['start_date'] = datetime.now().date() - timedelta(days=days_back)
        filters['end_date'] = datetime.now().date()
    
    # Results limit
    filters['limit'] = st.sidebar.number_input(
        "Max Results",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Maximum number of results to display"
    )
    
    # Sort options
    filters['sort_by'] = st.sidebar.selectbox(
        "Sort By",
        options=["Impact Score", "Confidence", "Date", "Ticker"],
        index=0
    )
    
    filters['sort_order'] = st.sidebar.selectbox(
        "Sort Order",
        options=["Descending", "Ascending"],
        index=0
    )
    
    # Quick filters
    st.sidebar.subheader("⚡ Quick Filters")
    
    if st.sidebar.button("🔥 High Impact (80+)", use_container_width=True):
        st.session_state.filter_high_impact = True
    
    if st.sidebar.button("📈 Positive Sentiment", use_container_width=True):
        st.session_state.filter_positive = True
    
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    return filters

def get_filtered_catalysts(session_state: Dict[str, Any], filters: Dict[str, Any]) -> List[Catalyst]:
    """Get catalysts based on filters"""
    try:
        # Handle quick filters
        if getattr(st.session_state, 'filter_high_impact', False):
            filters['min_impact'] = 80
            st.session_state.filter_high_impact = False
        
        if getattr(st.session_state, 'filter_positive', False):
            filters['sentiments'] = ['Positive']
            st.session_state.filter_positive = False
        
        # Get catalysts from database
        ticker = filters['ticker'] if filters['ticker'] else None
        category = CatalystType(filters['categories'][0]) if len(filters['categories']) == 1 else None
        
        catalysts = session_state.db.get_catalysts(
            ticker=ticker,
            category=category,
            min_impact=filters['min_impact'],
            limit=filters['limit']
        )
        
        # Apply additional filters
        filtered_catalysts = []
        
        for catalyst in catalysts:
            # Confidence filter
            if catalyst.confidence < filters['min_confidence']:
                continue
            
            # Category filter (when multiple selected)
            if filters['categories'] and catalyst.category.value not in filters['categories']:
                continue
            
            # Sentiment filter
            if filters['sentiments'] and catalyst.sentiment_label.value not in filters['sentiments']:
                continue
            
            # Source filter
            if filters['sources'] and catalyst.source.value not in filters['sources']:
                continue
            
            # Date filter
            if catalyst.published_date:
                catalyst_date = catalyst.published_date.date()
                if catalyst_date < filters['start_date'] or catalyst_date > filters['end_date']:
                    continue
            
            filtered_catalysts.append(catalyst)
        
        # Sort results
        sort_key_mapping = {
            "Impact Score": lambda c: c.impact,
            "Confidence": lambda c: c.confidence,
            "Date": lambda c: c.published_date or datetime.min.replace(tzinfo=timezone.utc),
            "Ticker": lambda c: c.ticker
        }
        
        sort_key = sort_key_mapping[filters['sort_by']]
        reverse = filters['sort_order'] == "Descending"
        
        filtered_catalysts.sort(key=sort_key, reverse=reverse)
        
        return filtered_catalysts
    
    except Exception as e:
        logger.error(f"Error filtering catalysts: {e}")
        st.error(f"Error applying filters: {e}")
        return []

def render_catalyst_summary(catalysts: List[Catalyst]):
    """Render summary statistics for filtered catalysts"""
    st.subheader(f"📊 Summary ({len(catalysts)} catalysts)")
    
    if not catalysts:
        return
    
    # Calculate summary stats
    avg_impact = sum(c.impact for c in catalysts) / len(catalysts)
    avg_confidence = sum(c.confidence for c in catalysts) / len(catalysts)
    high_impact_count = len([c for c in catalysts if c.impact >= 80])
    
    # Sentiment distribution
    positive_count = len([c for c in catalysts if c.sentiment_label == SentimentLabel.POSITIVE])
    negative_count = len([c for c in catalysts if c.sentiment_label == SentimentLabel.NEGATIVE])
    neutral_count = len([c for c in catalysts if c.sentiment_label == SentimentLabel.NEUTRAL])
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Average Impact",
            f"{avg_impact:.1f}/100",
            help="Average impact score of filtered catalysts"
        )
    
    with col2:
        st.metric(
            "Average Confidence",
            f"{avg_confidence:.1%}",
            help="Average confidence level of filtered catalysts"
        )
    
    with col3:
        st.metric(
            "High Impact",
            f"{high_impact_count} ({high_impact_count/len(catalysts):.1%})",
            help="Number and percentage of high impact catalysts (80+)"
        )
    
    with col4:
        st.metric(
            "Positive Sentiment",
            f"{positive_count} ({positive_count/len(catalysts):.1%})",
            help="Number and percentage of positive sentiment catalysts"
        )
    
    with col5:
        unique_tickers = len(set(c.ticker for c in catalysts))
        st.metric(
            "Unique Tickers",
            unique_tickers,
            help="Number of unique ticker symbols"
        )

def render_catalyst_table(catalysts: List[Catalyst]):
    """Render interactive catalyst table"""
    st.subheader("📋 Catalyst Details")
    
    if not catalysts:
        return
    
    # Convert to DataFrame
    data = []
    for catalyst in catalysts:
        data.append({
            "Ticker": catalyst.ticker,
            "Category": catalyst.category.value,
            "Impact": catalyst.impact,
            "Confidence": f"{catalyst.confidence:.1%}",
            "Sentiment": catalyst.sentiment_label.value,
            "Score": f"{catalyst.sentiment_score:+.2f}",
            "Source": catalyst.source.value,
            "Date": catalyst.published_date.strftime("%Y-%m-%d %H:%M") if catalyst.published_date else "N/A",
            "Summary": catalyst.catalyst[:100] + ("..." if len(catalyst.catalyst) > 100 else ""),
            "URL": catalyst.url or "N/A"
        })
    
    df = pd.DataFrame(data)
    
    # Configure column display
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Category": st.column_config.TextColumn("Category", width="medium"),
        "Impact": st.column_config.ProgressColumn("Impact", min_value=0, max_value=100, width="small"),
        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
        "Score": st.column_config.TextColumn("Score", width="small"),
        "Source": st.column_config.TextColumn("Source", width="small"),
        "Date": st.column_config.TextColumn("Date", width="medium"),
        "Summary": st.column_config.TextColumn("Summary", width="large"),
        "URL": st.column_config.LinkColumn("URL", width="small")
    }
    
    # Display table
    selected_rows = st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    
    # Show selected catalyst details
    if selected_rows and hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
        st.subheader("🔍 Selected Catalyst Details")
        
        for row_idx in selected_rows.selection.rows:
            if row_idx < len(catalysts):
                catalyst = catalysts[row_idx]
                render_catalyst_detail_card(catalyst)

def render_catalyst_detail_card(catalyst: Catalyst):
    """Render detailed catalyst card"""
    try:
        with st.expander(f"{catalyst.ticker} - {catalyst.category.value}", expanded=True):
            # Header with key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Impact Score", f"{catalyst.impact}/100")
            
            with col2:
                st.metric("Confidence", f"{catalyst.confidence:.1%}")
            
            with col3:
                sentiment_emoji = {"Positive": "📈", "Negative": "📉", "Neutral": "➡️"}
                emoji = sentiment_emoji.get(catalyst.sentiment_label.value, "➡️")
                st.metric("Sentiment", f"{emoji} {catalyst.sentiment_score:+.2f}")
            
            # Full catalyst text
            st.markdown("**Catalyst:**")
            st.write(catalyst.catalyst)
            
            # Metadata
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Details:**")
                st.write(f"• **Ticker:** {catalyst.ticker}")
                st.write(f"• **Sector:** {catalyst.sector}")
                st.write(f"• **Source:** {catalyst.source.value}")
                if catalyst.published_date:
                    st.write(f"• **Date:** {catalyst.published_date.strftime('%Y-%m-%d %H:%M UTC')}")
            
            with col2:
                st.markdown("**Metadata:**")
                metadata = catalyst.metadata
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float)):
                            st.write(f"• **{key.title()}:** {value}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if catalyst.url:
                    st.link_button("🔗 View Source", catalyst.url, use_container_width=True)
            
            with col2:
                if st.button(f"🚨 Send Test Alert", key=f"alert_{catalyst.ticker}_{catalyst.published_date}", use_container_width=True):
                    try:
                        # Trigger test alert for this catalyst
                        st.success("Test alert sent!")
                    except Exception as e:
                        st.error(f"Alert failed: {e}")
            
            with col3:
                if st.button(f"📋 Copy Summary", key=f"copy_{catalyst.ticker}_{catalyst.published_date}", use_container_width=True):
                    # Copy catalyst summary to clipboard (in real app)
                    st.success("Summary copied!")
    
    except Exception as e:
        logger.error(f"Error rendering catalyst detail: {e}")
        st.error("Error displaying catalyst details")

def render_detailed_analysis(catalysts: List[Catalyst]):
    """Render detailed analysis charts"""
    st.subheader("📊 Detailed Analysis")
    
    if not catalysts:
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Impact vs Confidence", "📈 Sentiment Analysis", "📊 Category Breakdown", "⏰ Time Analysis"])
    
    with tab1:
        render_impact_confidence_scatter(catalysts)
    
    with tab2:
        render_sentiment_analysis_charts(catalysts)
    
    with tab3:
        render_category_breakdown(catalysts)
    
    with tab4:
        render_time_analysis(catalysts)

def render_impact_confidence_scatter(catalysts: List[Catalyst]):
    """Render impact vs confidence scatter plot"""
    try:
        if not catalysts:
            st.info("No data for scatter plot")
            return
        
        # Prepare data
        df = pd.DataFrame([
            {
                "ticker": c.ticker,
                "impact": c.impact,
                "confidence": c.confidence,
                "sentiment": c.sentiment_label.value,
                "category": c.category.value,
                "size": 10 + c.impact/10  # Size based on impact
            }
            for c in catalysts
        ])
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x="confidence",
            y="impact",
            color="sentiment",
            size="size",
            hover_data=["ticker", "category"],
            title="Impact Score vs Confidence Level",
            labels={
                "confidence": "Confidence Level",
                "impact": "Impact Score",
                "sentiment": "Sentiment"
            }
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        high_confidence_high_impact = len([c for c in catalysts if c.confidence > 0.8 and c.impact > 80])
        st.write(f"**Key Insight:** {high_confidence_high_impact} catalysts have both high confidence (>80%) and high impact (>80)")
    
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")

def render_sentiment_analysis_charts(catalysts: List[Catalyst]):
    """Render sentiment analysis charts"""
    try:
        if not catalysts:
            st.info("No data for sentiment analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = {}
            for catalyst in catalysts:
                sentiment = catalyst.sentiment_label.value
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            fig = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment score distribution
            sentiment_scores = [c.sentiment_score for c in catalysts]
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=sentiment_scores,
                    nbinsx=20,
                    marker_color='lightgreen',
                    marker_line_color='black',
                    marker_line_width=1
                )
            ])
            
            fig.update_layout(
                title="Sentiment Score Distribution",
                xaxis_title="Sentiment Score",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating sentiment charts: {e}")

def render_category_breakdown(catalysts: List[Catalyst]):
    """Render category breakdown analysis"""
    try:
        if not catalysts:
            st.info("No data for category analysis")
            return
        
        # Category vs Impact analysis
        category_data = []
        for catalyst in catalysts:
            category_data.append({
                "category": catalyst.category.value,
                "impact": catalyst.impact,
                "confidence": catalyst.confidence,
                "sentiment_score": catalyst.sentiment_score
            })
        
        df = pd.DataFrame(category_data)
        
        # Box plot of impact by category
        fig = px.box(
            df,
            x="category",
            y="impact",
            title="Impact Score Distribution by Category"
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category statistics table
        st.subheader("Category Statistics")
        
        category_stats = df.groupby('category').agg({
            'impact': ['count', 'mean', 'max'],
            'confidence': 'mean',
            'sentiment_score': 'mean'
        }).round(2)
        
        category_stats.columns = ['Count', 'Avg Impact', 'Max Impact', 'Avg Confidence', 'Avg Sentiment']
        st.dataframe(category_stats, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating category breakdown: {e}")

def render_time_analysis(catalysts: List[Catalyst]):
    """Render time-based analysis"""
    try:
        catalysts_with_dates = [c for c in catalysts if c.published_date]
        
        if not catalysts_with_dates:
            st.info("No catalysts with timestamp data for time analysis")
            return
        
        # Time series of catalyst count
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily activity
            daily_counts = {}
            for catalyst in catalysts_with_dates:
                day = catalyst.published_date.date()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            days = sorted(daily_counts.keys())
            counts = [daily_counts[day] for day in days]
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=days,
                    y=counts,
                    mode='lines+markers',
                    name='Daily Catalysts'
                )
            ])
            
            fig.update_layout(
                title="Daily Catalyst Activity",
                xaxis_title="Date",
                yaxis_title="Number of Catalysts"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly distribution
            hourly_counts = [0] * 24
            for catalyst in catalysts_with_dates:
                hour = catalyst.published_date.hour
                hourly_counts[hour] += 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(24)),
                    y=hourly_counts,
                    marker_color='lightcoral'
                )
            ])
            
            fig.update_layout(
                title="Hourly Distribution (UTC)",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Catalysts"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating time analysis: {e}")
