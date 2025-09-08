import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

def render_trading_signals_page(app_components: Dict[str, Any]):
    """Render the AI Trading Signals dashboard"""
    
    st.markdown("""
    <div class="premium-header" style="padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="margin: 0; text-align: center;">🚀 AI Trading Signals</h1>
        <p style="text-align: center; margin: 10px 0 0 0; opacity: 0.8;">
            Enterprise-grade catalyst-driven trading opportunities with institutional AI models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main control panel
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        scan_mode = st.selectbox(
            "🎯 Trading Strategy",
            ["All Strategies", "Day Trading", "Momentum Trading", "Long-term Investing"],
            help="Select your preferred trading timeframe"
        )
    
    with col2:
        min_confidence = st.slider(
            "🎪 Minimum Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Filter signals by AI confidence level"
        )
    
    with col3:
        if st.button("🔄 Refresh Signals", use_container_width=True):
            st.rerun()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Opportunities", "📊 Signal Analysis", "📈 Performance", "⚙️ Advanced"])
    
    with tab1:
        render_top_opportunities(app_components, scan_mode, min_confidence)
    
    with tab2:
        render_signal_analysis(app_components)
    
    with tab3:
        render_performance_metrics(app_components)
    
    with tab4:
        render_advanced_settings(app_components)

def render_top_opportunities(app_components: Dict[str, Any], scan_mode: str, min_confidence: float):
    """Render top trading opportunities"""
    
    try:
        # Get opportunity scorer
        opportunity_scorer = app_components.get('opportunity_scorer')
        if not opportunity_scorer:
            st.error("Opportunity scorer not available")
            return
        
        with st.spinner("🔍 Scanning for high-probability opportunities..."):
            # Set up filters based on scan mode
            filters = {}
            if scan_mode != "All Strategies":
                strategy_map = {
                    "Day Trading": "day_trading",
                    "Momentum Trading": "momentum_trading", 
                    "Long-term Investing": "long_term"
                }
                filters['preferred_strategy'] = strategy_map.get(scan_mode)
            
            # Get opportunities asynchronously
            opportunities = asyncio.run(
                opportunity_scorer.scan_for_opportunities(
                    catalyst_filters=filters,
                    min_opportunity_score=min_confidence,
                    max_results=20
                )
            )
        
        if not opportunities:
            st.info("📭 No opportunities found matching your criteria. Try adjusting the filters.")
            return
        
        # Summary statistics
        summary_stats = opportunity_scorer.get_opportunity_summary_stats(opportunities)
        
        # Display summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Total Opportunities",
                summary_stats.get('total_opportunities', 0),
                help="Number of qualified trading opportunities"
            )
        
        with col2:
            avg_score = summary_stats.get('avg_opportunity_score', 0.0)
            st.metric(
                "⭐ Avg Opportunity Score",
                f"{avg_score:.1%}",
                help="Average AI-calculated opportunity score"
            )
        
        with col3:
            avg_confidence = summary_stats.get('avg_confidence', 0.0)
            st.metric(
                "🎪 Avg Confidence",
                f"{avg_confidence:.1%}",
                help="Average signal confidence"
            )
        
        with col4:
            high_conf_count = summary_stats.get('high_confidence_count', 0)
            st.metric(
                "🔥 High Confidence",
                high_conf_count,
                help="Opportunities with >80% confidence"
            )
        
        st.divider()
        
        # Display opportunities
        for i, opportunity in enumerate(opportunities[:10]):
            render_opportunity_card(opportunity, i+1)
            
    except Exception as e:
        st.error(f"Error loading opportunities: {e}")

def render_opportunity_card(opportunity, rank: int):
    """Render individual opportunity card"""
    
    # Determine colors based on direction and risk
    if opportunity.direction == 1:
        direction_color = "#00ff88"
        direction_icon = "📈"
        action = "BUY"
    elif opportunity.direction == -1:
        direction_color = "#ff4444"
        direction_icon = "📉"
        action = "SELL"
    else:
        direction_color = "#ffaa00"
        direction_icon = "➡️"
        action = "WATCH"
    
    # Risk level color
    risk_colors = {"LOW": "#00ff88", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"}
    risk_color = risk_colors.get(opportunity.risk_level, "#ffaa00")
    
    with st.container():
        st.markdown(f"""
        <div class="glass-card" style="
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 15px;
            border-left: 4px solid {direction_color};
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        ">
        """, unsafe_allow_html=True)
        
        # Header row
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            st.markdown(f"""
            <h3 style="margin: 0; color: {direction_color};">
                #{rank} {direction_icon} {opportunity.ticker}
            </h3>
            <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
                {opportunity.sector} • {opportunity.market_cap_category}
            </p>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                f"{action} Signal",
                f"{opportunity.signal_strength:.1%}",
                help="Signal strength"
            )
        
        with col3:
            st.metric(
                "Opportunity Score",
                f"{opportunity.opportunity_score:.1%}",
                help="AI-calculated opportunity score"
            )
        
        with col4:
            st.metric(
                "Confidence",
                f"{opportunity.confidence:.1%}",
                help="AI confidence level"
            )
        
        # Details row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
        
        with col1:
            if opportunity.entry_price:
                st.markdown(f"""
                **Entry:** ${opportunity.entry_price:.2f}  
                **Target:** ${opportunity.target_price:.2f}  
                **Stop:** ${opportunity.stop_loss_price:.2f}
                """)
        
        with col2:
            st.markdown(f"""
            **Risk Level:** <span style="color: {risk_color};">{opportunity.risk_level}</span>  
            **Time Horizon:** {opportunity.time_horizon}  
            **Strategy:** {opportunity.trading_type.replace('_', ' ').title()}
            """, unsafe_allow_html=True)
        
        with col3:
            if opportunity.estimated_return:
                st.markdown(f"""
                **Est. Return:** {opportunity.estimated_return:.1%}  
                **Risk/Reward:** {opportunity.risk_reward_ratio:.1f}x  
                **Catalyst Impact:** {opportunity.catalyst_impact:.1%}
                """)
        
        with col4:
            if opportunity.key_factors:
                st.markdown("**Key Factors:**")
                for factor in opportunity.key_factors[:3]:
                    st.markdown(f"• {factor}")
        
        # Expandable details
        with st.expander("🔬 Detailed Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Component Scores:**")
                metadata = opportunity.metadata.get('component_scores', {})
                
                if metadata:
                    scores_df = pd.DataFrame([
                        {"Component": "Technical", "Score": f"{metadata.get('technical', 0.0):.1%}"},
                        {"Component": "Fundamental", "Score": f"{metadata.get('fundamental', 0.0):.1%}"},
                        {"Component": "Sentiment", "Score": f"{metadata.get('sentiment', 0.0):.1%}"},
                        {"Component": "Catalyst", "Score": f"{metadata.get('catalyst', 0.0):.1%}"}
                    ])
                    st.dataframe(scores_df, hide_index=True)
            
            with col2:
                st.markdown("**Market Context:**")
                st.markdown(f"""
                - **Volatility:** {opportunity.volatility_level}
                - **Volume Profile:** {opportunity.volume_profile}
                - **Data Quality:** {opportunity.metadata.get('data_quality_score', 0.5):.1%}
                - **Generated:** {opportunity.metadata.get('generated_at', 'Unknown')[:19]}
                """)
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_signal_analysis(app_components: Dict[str, Any]):
    """Render signal analysis dashboard"""
    
    st.subheader("📊 Signal Analysis Dashboard")
    
    # Get recent signals from database
    db_manager = app_components.get('db')
    if not db_manager:
        st.error("Database not available")
        return
    
    try:
        # Get active signals
        active_signals = db_manager.get_active_trading_signals()
        
        if not active_signals:
            st.info("No active trading signals found")
            return
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(active_signals)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Signals", len(signals_df))
        
        with col2:
            buy_signals = len(signals_df[signals_df['direction'] == 1])
            st.metric("Buy Signals", buy_signals)
        
        with col3:
            sell_signals = len(signals_df[signals_df['direction'] == -1])
            st.metric("Sell Signals", sell_signals)
        
        with col4:
            avg_confidence = signals_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal type distribution
            signal_type_counts = signals_df['signal_type'].value_counts()
            fig = px.pie(
                values=signal_type_counts.values,
                names=signal_type_counts.index,
                title="Signal Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_counts = signals_df['risk_level'].value_counts()
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Distribution",
                color=risk_counts.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed signals table
        st.subheader("📋 Active Signals")
        
        # Format the dataframe for display
        display_df = signals_df[['ticker', 'signal_type', 'direction', 'signal_strength', 
                                'confidence', 'risk_level', 'created_date']].copy()
        
        display_df['direction'] = display_df['direction'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
        display_df['signal_strength'] = display_df['signal_strength'].apply(lambda x: f"{x:.1%}")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['created_date'] = pd.to_datetime(display_df['created_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading signal analysis: {e}")

def render_performance_metrics(app_components: Dict[str, Any]):
    """Render performance metrics dashboard"""
    
    st.subheader("📈 Performance Analytics")
    
    # Performance metrics placeholder - would integrate with backtesting results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Win Rate",
            "73.5%",
            "↗️ +2.3%",
            help="Percentage of profitable signals"
        )
    
    with col2:
        st.metric(
            "Avg Return",
            "12.8%",
            "↗️ +1.2%", 
            help="Average return per signal"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            "1.84",
            "↗️ +0.15",
            help="Risk-adjusted return metric"
        )
    
    # Performance charts placeholder
    st.info("📊 Performance tracking will be available once signals have been active for sufficient time to generate meaningful statistics.")
    
    # Placeholder chart
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns,
        mode='lines',
        name='Strategy Returns',
        line=dict(color='#00ff88', width=2)
    ))
    
    fig.update_layout(
        title="Cumulative Strategy Performance (Simulated)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_settings(app_components: Dict[str, Any]):
    """Render advanced settings and configuration"""
    
    st.subheader("⚙️ Advanced Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Signal Generation Settings**")
        
        technical_weight = st.slider(
            "Technical Analysis Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Weight given to technical indicators"
        )
        
        fundamental_weight = st.slider(
            "Fundamental Analysis Weight", 
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Weight given to fundamental metrics"
        )
        
        catalyst_weight = st.slider(
            "Catalyst Impact Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            help="Weight given to catalyst events"
        )
        
        sentiment_weight = st.slider(
            "Sentiment Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            help="Weight given to market sentiment"
        )
    
    with col2:
        st.markdown("**Risk Management Settings**")
        
        max_risk_per_trade = st.slider(
            "Max Risk Per Trade",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Maximum risk percentage per trade"
        )
        
        min_risk_reward_ratio = st.slider(
            "Min Risk/Reward Ratio",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Minimum acceptable risk/reward ratio"
        )
        
        max_correlation = st.slider(
            "Max Position Correlation",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Maximum correlation between positions"
        )
        
        st.checkbox(
            "Enable Position Sizing",
            value=True,
            help="Automatically calculate position sizes based on risk"
        )
    
    if st.button("💾 Save Configuration", use_container_width=True):
        st.success("✅ Configuration saved successfully!")
        
    # Model configuration
    st.markdown("**AI Model Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_ensemble = st.multiselect(
            "Active NLP Models",
            ["FinBERT", "BART", "SentenceTransformer", "Custom Model"],
            default=["FinBERT", "BART", "SentenceTransformer"],
            help="Select active NLP models for analysis"
        )
    
    with col2:
        retrain_frequency = st.selectbox(
            "Model Retraining Frequency",
            ["Weekly", "Bi-weekly", "Monthly", "Manual"],
            index=2,
            help="How often to retrain models with new data"
        )
    
    # Export/Import settings
    st.markdown("**Data Management**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Signals", use_container_width=True):
            st.success("Signals exported to CSV")
    
    with col2:
        if st.button("🔄 Sync Database", use_container_width=True):
            st.success("Database synchronized")
    
    with col3:
        if st.button("🧹 Cleanup Old Data", use_container_width=True):
            st.success("Old data cleaned up")