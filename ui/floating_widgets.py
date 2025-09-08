"""
Floating Widget Components for Glassmorphism UI
HUD-style trading interface elements
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

def render_catalyst_alert_widget(catalysts: List[Any], max_items: int = 3):
    """Render floating catalyst alerts widget"""
    
    widget_html = f"""
    <div class="floating-widget">
        <div class="glass-title" style="font-size: 1.2rem; margin-bottom: 1rem;">
            🎯 Live Catalyst Alerts
            <span class="status-indicator status-online"></span>
        </div>
    """
    
    if catalysts:
        for i, catalyst in enumerate(catalysts[:max_items]):
            confidence_color = "#10b981" if catalyst.confidence > 0.8 else "#f59e0b" if catalyst.confidence > 0.6 else "#ef4444"
            
            widget_html += f"""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
                border-left: 3px solid {confidence_color};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: var(--text-primary);">{catalyst.ticker}</span>
                    <span style="
                        background: {confidence_color};
                        color: white;
                        padding: 0.2rem 0.5rem;
                        border-radius: 12px;
                        font-size: 0.8rem;
                        font-family: 'JetBrains Mono', monospace;
                    ">{catalyst.confidence:.0%}</span>
                </div>
                <div style="
                    color: var(--text-secondary);
                    font-size: 0.9rem;
                    margin-top: 0.25rem;
                    line-height: 1.3;
                ">
                    {catalyst.catalyst[:80]}...
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-top: 0.5rem;
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                ">
                    <span>{catalyst.category.value}</span>
                    <span>{catalyst.published_date.strftime('%H:%M')}</span>
                </div>
            </div>
            """
    else:
        widget_html += """
        <div style="
            text-align: center;
            color: var(--text-secondary);
            padding: 2rem 1rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📡</div>
            <div>Monitoring for catalysts...</div>
        </div>
        """
    
    widget_html += "</div>"
    
    st.markdown(widget_html, unsafe_allow_html=True)

def render_ai_signal_widget(signals: List[Any], max_items: int = 2):
    """Render AI trading signals floating widget"""
    
    widget_html = f"""
    <div class="floating-widget hud-element">
        <div class="glass-title" style="font-size: 1.2rem; margin-bottom: 1rem;">
            🤖 AI Trading Signals
        </div>
    """
    
    if signals:
        for signal in signals[:max_items]:
            signal_color = {
                'buy': '#10b981',
                'strong_buy': '#059669',
                'sell': '#ef4444',
                'strong_sell': '#dc2626',
                'hold': '#3b82f6'
            }.get(signal.signal_type.value, '#6b7280')
            
            widget_html += f"""
            <div class="signal-card {signal.signal_type.value}" style="
                margin: 0.75rem 0;
                padding: 1rem;
                background: linear-gradient(145deg, var(--glass-bg), {signal_color}15);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.1rem; color: var(--text-primary);">
                            {signal.ticker}
                        </span>
                        <span style="
                            background: {signal_color};
                            color: white;
                            padding: 0.2rem 0.6rem;
                            border-radius: 12px;
                            font-size: 0.8rem;
                            margin-left: 0.5rem;
                            font-weight: 600;
                        ">
                            {signal.signal_type.value.replace('_', ' ').upper()}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600; color: var(--text-primary);">
                            {signal.confidence:.0%}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            confidence
                        </div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; margin: 0.75rem 0;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">Entry</div>
                        <div style="font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            ${signal.entry_price:.2f}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">Target</div>
                        <div style="font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            ${signal.target_price:.2f}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">R/R</div>
                        <div style="font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            {signal.risk_reward_ratio:.1f}:1
                        </div>
                    </div>
                </div>
                
                <div style="
                    font-size: 0.85rem;
                    color: var(--text-secondary);
                    border-top: 1px solid var(--glass-border);
                    padding-top: 0.5rem;
                    margin-top: 0.5rem;
                ">
                    {signal.reasoning[0] if signal.reasoning else 'AI-generated signal'}
                </div>
            </div>
            """
    else:
        widget_html += """
        <div style="
            text-align: center;
            color: var(--text-secondary);
            padding: 2rem 1rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
            <div>Analyzing market conditions...</div>
        </div>
        """
    
    widget_html += "</div>"
    
    st.markdown(widget_html, unsafe_allow_html=True)

def render_market_metrics_widget(metrics: Dict[str, Any]):
    """Render market metrics floating widget"""
    
    widget_html = """
    <div class="floating-widget">
        <div class="glass-title" style="font-size: 1.2rem; margin-bottom: 1rem;">
            📊 Market Pulse
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
    """
    
    default_metrics = {
        'Active Catalysts': {'value': 0, 'change': 0, 'color': '#3b82f6'},
        'Signal Accuracy': {'value': 0.75, 'change': 0.02, 'color': '#10b981'},
        'Processing Speed': {'value': 2.4, 'change': -0.1, 'color': '#8b5cf6'},
        'Market Volatility': {'value': 0.18, 'change': 0.03, 'color': '#f59e0b'}
    }
    
    metrics = {**default_metrics, **metrics}
    
    for metric_name, data in metrics.items():
        value = data['value']
        change = data['change']
        color = data['color']
        
        # Format value based on type
        if isinstance(value, float):
            if metric_name == 'Signal Accuracy':
                display_value = f"{value:.0%}"
            elif metric_name == 'Processing Speed':
                display_value = f"{value:.1f}s"
            elif metric_name == 'Market Volatility':
                display_value = f"{value:.1%}"
            else:
                display_value = f"{value:.2f}"
        else:
            display_value = str(value)
        
        change_color = '#10b981' if change >= 0 else '#ef4444'
        change_symbol = '+' if change >= 0 else ''
        
        widget_html += f"""
        <div class="metric-glass" style="border-left: 3px solid {color};">
            <div style="
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-primary);
                font-family: 'JetBrains Mono', monospace;
                margin-bottom: 0.25rem;
            ">
                {display_value}
            </div>
            <div style="
                font-size: 0.9rem;
                color: var(--text-secondary);
                margin-bottom: 0.25rem;
            ">
                {metric_name}
            </div>
            <div style="
                font-size: 0.8rem;
                color: {change_color};
                font-weight: 500;
            ">
                {change_symbol}{change:.2f} today
            </div>
        </div>
        """
    
    widget_html += """
        </div>
    </div>
    """
    
    st.markdown(widget_html, unsafe_allow_html=True)

def render_system_status_widget(components: Dict[str, str]):
    """Render system status HUD widget"""
    
    widget_html = """
    <div class="floating-widget hud-element">
        <div class="glass-title" style="font-size: 1.2rem; margin-bottom: 1rem;">
            ⚡ System Status
        </div>
    """
    
    default_components = {
        'WebSocket Feeds': 'online',
        'AI Catalyst Detection': 'online',
        'Signal Generator': 'online',
        'Data Processing': 'warning',
        'Alert System': 'online',
        'Database': 'online'
    }
    
    components = {**default_components, **components}
    
    for component, status in components.items():
        status_color = {
            'online': '#10b981',
            'warning': '#f59e0b',
            'offline': '#ef4444'
        }.get(status, '#6b7280')
        
        widget_html += f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--glass-border);
        ">
            <span style="color: var(--text-primary); font-weight: 500;">
                {component}
            </span>
            <div style="display: flex; align-items: center;">
                <span class="status-indicator" style="
                    background: {status_color};
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    margin-right: 0.5rem;
                    box-shadow: 0 0 6px {status_color};
                "></span>
                <span style="
                    color: {status_color};
                    font-size: 0.9rem;
                    font-weight: 500;
                    text-transform: uppercase;
                ">
                    {status}
                </span>
            </div>
        </div>
        """
    
    widget_html += "</div>"
    
    st.markdown(widget_html, unsafe_allow_html=True)

def render_mini_chart_widget(ticker: str, data: List[Dict]):
    """Render mini price chart widget"""
    
    if not data:
        # Generate demo data
        dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        prices = [100 + i * 0.5 + (i % 3) * 2 for i in range(24)]
        data = [{'time': d, 'price': p} for d, p in zip(dates, prices)]
    
    # Create mini chart
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['price'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)',
        name=ticker
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False
    )
    
    widget_html = f"""
    <div class="floating-widget" style="padding: 1rem;">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        ">
            <span style="font-weight: 600; color: var(--text-primary);">
                {ticker}
            </span>
            <span style="
                font-family: 'JetBrains Mono', monospace;
                font-weight: 600;
                color: var(--accent-green);
            ">
                ${data[-1]['price']:.2f}
            </span>
        </div>
    </div>
    """
    
    st.markdown(widget_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")

def render_floating_widgets_grid(session_state):
    """Render grid of floating widgets"""
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Get recent catalysts
        catalysts = []
        if hasattr(session_state, 'catalyst_manager'):
            try:
                catalysts = session_state.catalyst_manager.get_recent_catalysts(limit=3)
            except:
                pass
        
        render_catalyst_alert_widget(catalysts)
        
        # System status
        render_system_status_widget({})
    
    with col2:
        # Trading signals
        signals = []
        if hasattr(session_state, 'automated_signal_generator'):
            try:
                signals = session_state.automated_signal_generator.get_active_signals()
            except:
                pass
        
        render_ai_signal_widget(signals)
        
        # Mini chart
        render_mini_chart_widget("AAPL", [])
    
    with col3:
        # Market metrics
        render_market_metrics_widget({})
        
        # Another chart or widget
        render_mini_chart_widget("TSLA", [])