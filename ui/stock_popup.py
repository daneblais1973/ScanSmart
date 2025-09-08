import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import asyncio
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_company_info(ticker: str) -> Dict[str, Any]:
    """Get company information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'website': info.get('website', ''),
            'description': info.get('longBusinessSummary', ''),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A'),
            'country': info.get('country', 'N/A'),
            'city': info.get('city', 'N/A'),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'price_to_book': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0)
        }
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A'}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_news(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent news for a stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        formatted_news = []
        for article in news[:limit]:
            formatted_news.append({
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'publisher': article.get('publisher', ''),
                'publish_time': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                'summary': article.get('summary', '')
            })
        
        return formatted_news
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

def create_stock_chart(ticker: str, period: str = '3mo', selected_indicators: Dict[str, set] = None) -> go.Figure:
    """Create an interactive stock chart with technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{ticker} Stock Price', 'Volume', 'Technical Indicators'),
            row_width=[0.7, 0.2, 0.1]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add selected technical indicators
        if selected_indicators:
            add_technical_indicators_to_chart(fig, data, selected_indicators)
        
        # Add volume
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} - Technical Analysis Chart',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update x-axis
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Indicators", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating chart for {ticker}: {e}")
        return None

def add_technical_indicators_to_chart(fig: go.Figure, data: pd.DataFrame, selected_indicators: Dict[str, set]):
    """Add selected technical indicators to the chart"""
    try:
        # Moving Averages
        if 'trend' in selected_indicators:
            if 'SMA_20' in selected_indicators['trend']:
                sma_20 = data['Close'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
            
            if 'SMA_50' in selected_indicators['trend']:
                sma_50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_50, name='SMA 50', line=dict(color='blue')),
                    row=1, col=1
                )
            
            if 'EMA_12' in selected_indicators['trend']:
                ema_12 = data['Close'].ewm(span=12).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=ema_12, name='EMA 12', line=dict(color='yellow')),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if 'volatility' in selected_indicators and 'Bollinger_Bands' in selected_indicators['volatility']:
            bb_period = 20
            bb_std = 2
            sma = data['Close'].rolling(window=bb_period).mean()
            std = data['Close'].rolling(window=bb_period).std()
            upper_band = sma + (std * bb_std)
            lower_band = sma - (std * bb_std)
            
            fig.add_trace(
                go.Scatter(x=data.index, y=upper_band, name='BB Upper', 
                          line=dict(color='purple', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=lower_band, name='BB Lower',
                          line=dict(color='purple', dash='dash'),
                          fill='tonexty', fillcolor='rgba(128, 0, 128, 0.1)'),
                row=1, col=1
            )
        
        # RSI
        if 'momentum' in selected_indicators and 'RSI' in selected_indicators['momentum']:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='cyan')),
                row=3, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'trend' in selected_indicators and 'MACD' in selected_indicators['trend']:
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            fig.add_trace(
                go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=signal, name='MACD Signal', line=dict(color='red')),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=data.index, y=histogram, name='MACD Histogram', opacity=0.6),
                row=3, col=1
            )
        
    except Exception as e:
        logger.error(f"Error adding technical indicators to chart: {e}")

def render_stock_popup(ticker: str, catalyst_data: Dict = None, session_state = None):
    """Render the comprehensive stock detail popup"""
    
    st.markdown(f"# 📊 {ticker} - Detailed Analysis")
    
    # Get company info and news
    company_info = get_company_info(ticker)
    news_data = get_stock_news(ticker)
    
    # Create layout with columns
    top_row = st.columns([1, 1])
    
    # ============================================================================ 
    # TOP LEFT: Company Details
    # ============================================================================
    with top_row[0]:
        st.markdown("### 🏢 Company Details")
        
        with st.container():
            st.markdown(f"**{company_info['name']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sector", company_info['sector'])
                st.metric("Industry", company_info['industry'])
                st.metric("Exchange", company_info['exchange'])
                
            with col2:
                if company_info['market_cap']:
                    market_cap_b = company_info['market_cap'] / 1e9
                    st.metric("Market Cap", f"${market_cap_b:.2f}B")
                
                if company_info['employees']:
                    st.metric("Employees", f"{company_info['employees']:,}")
                
                if company_info['pe_ratio']:
                    st.metric("P/E Ratio", f"{company_info['pe_ratio']:.2f}")
            
            # Key metrics
            st.markdown("**Key Metrics:**")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                if company_info['dividend_yield']:
                    st.write(f"💰 Dividend Yield: {company_info['dividend_yield']*100:.2f}%")
                if company_info['beta']:
                    st.write(f"📈 Beta: {company_info['beta']:.2f}")
            
            with metrics_col2:
                if company_info['price_to_book']:
                    st.write(f"📊 P/B Ratio: {company_info['price_to_book']:.2f}")
                if company_info['forward_pe']:
                    st.write(f"🔮 Forward P/E: {company_info['forward_pe']:.2f}")
    
    # ============================================================================
    # TOP RIGHT: Buy/Sell Signals & Reasoning  
    # ============================================================================
    with top_row[1]:
        st.markdown("### 🎯 Trading Signals")
        
        # Generate signals based on available data
        signals = generate_trading_signals(ticker, catalyst_data)
        
        # Display overall signal
        if signals['overall_signal'] > 0.6:
            st.success("🟢 **STRONG BUY**")
            signal_color = "green"
        elif signals['overall_signal'] > 0.3:
            st.info("🔵 **BUY**")
            signal_color = "blue" 
        elif signals['overall_signal'] > -0.3:
            st.warning("🟡 **HOLD**")
            signal_color = "orange"
        elif signals['overall_signal'] > -0.6:
            st.error("🔴 **SELL**")
            signal_color = "red"
        else:
            st.error("🔴 **STRONG SELL**")
            signal_color = "darkred"
        
        # Signal strength gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=signals['overall_signal'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Signal Strength"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': signal_color},
                'steps': [
                    {'range': [-100, -60], 'color': "red"},
                    {'range': [-60, -30], 'color': "orange"},
                    {'range': [-30, 30], 'color': "yellow"},
                    {'range': [30, 60], 'color': "lightgreen"},
                    {'range': [60, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Signal breakdown
        st.markdown("**Signal Breakdown:**")
        for signal_type, strength in signals['signal_breakdown'].items():
            if strength > 0.5:
                emoji = "🟢"
            elif strength > 0:
                emoji = "🔵"
            elif strength > -0.5:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            st.write(f"{emoji} {signal_type.replace('_', ' ').title()}: {strength:.2f}")
        
        # Reasoning
        st.markdown("**Key Reasoning:**")
        for reason in signals['reasoning'][:3]:  # Top 3 reasons
            st.write(f"• {reason}")
    
    # ============================================================================
    # CENTER: Related News
    # ============================================================================
    st.markdown("### 📰 Related News & Analysis")
    
    if news_data:
        # Display news in expandable format
        for i, article in enumerate(news_data[:5]):  # Show top 5 articles
            with st.expander(f"📄 {article['title']}", expanded=(i == 0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if article['summary']:
                        st.write(article['summary'])
                    else:
                        st.write("Summary not available")
                
                with col2:
                    st.write(f"**Publisher:** {article['publisher']}")
                    st.write(f"**Published:** {article['publish_time'].strftime('%Y-%m-%d %H:%M')}")
                    if article['link']:
                        st.markdown(f"[Read Full Article]({article['link']})")
    else:
        st.info("No recent news available for this stock")
    
    # Add catalyst information if available
    if catalyst_data:
        st.markdown("### 🚀 Catalyst Information")
        st.json(catalyst_data)
    
    # ============================================================================
    # BOTTOM: Interactive Chart with Indicator Selection
    # ============================================================================
    st.markdown("### 📈 Technical Analysis Chart")
    
    # Chart controls
    chart_col1, chart_col2, chart_col3 = st.columns([1, 1, 2])
    
    with chart_col1:
        period = st.selectbox(
            "Time Period",
            options=['1mo', '3mo', '6mo', '1y', '2y'],
            index=1,
            key="chart_period"
        )
    
    with chart_col2:
        interval = st.selectbox(
            "Interval", 
            options=['1d', '1wk', '1mo'],
            index=0,
            key="chart_interval"
        )
    
    with chart_col3:
        # Quick indicator selection
        st.markdown("**Quick Add Indicators:**")
        quick_cols = st.columns(4)
        
        with quick_cols[0]:
            add_sma = st.checkbox("SMA 20/50", key="quick_sma")
        with quick_cols[1]:
            add_bb = st.checkbox("Bollinger Bands", key="quick_bb")
        with quick_cols[2]:
            add_rsi = st.checkbox("RSI", key="quick_rsi")
        with quick_cols[3]:
            add_macd = st.checkbox("MACD", key="quick_macd")
    
    # Build selected indicators for chart
    chart_indicators = {'trend': set(), 'momentum': set(), 'volatility': set()}
    
    if add_sma:
        chart_indicators['trend'].update(['SMA_20', 'SMA_50'])
    if add_bb:
        chart_indicators['volatility'].add('Bollinger_Bands')
    if add_rsi:
        chart_indicators['momentum'].add('RSI')
    if add_macd:
        chart_indicators['trend'].add('MACD')
    
    # Use session indicators if available
    if session_state and hasattr(session_state, 'selected_indicators'):
        for category, indicators in session_state.selected_indicators.items():
            if category in chart_indicators:
                chart_indicators[category].update(indicators)
    
    # Create and display chart
    chart = create_stock_chart(ticker, period, chart_indicators)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.error("Unable to load chart data")
    
    # Chart indicator legend
    if any(chart_indicators.values()):
        st.markdown("**Active Indicators:**")
        for category, indicators in chart_indicators.items():
            if indicators:
                st.write(f"• **{category.title()}:** {', '.join(indicators)}")

def generate_trading_signals(ticker: str, catalyst_data: Dict = None) -> Dict[str, Any]:
    """Generate trading signals for the stock"""
    try:
        # Get recent stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period='3mo')
        
        if data.empty:
            return {
                'overall_signal': 0,
                'signal_breakdown': {},
                'reasoning': ['Insufficient data for analysis']
            }
        
        signals = {}
        reasoning = []
        
        # Technical signals
        current_price = data['Close'].iloc[-1]
        
        # Moving average signals
        if len(data) >= 50:
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            if current_price > sma_20 > sma_50:
                signals['trend_signal'] = 0.8
                reasoning.append("Price above both SMA20 and SMA50 - Strong uptrend")
            elif current_price > sma_20:
                signals['trend_signal'] = 0.4
                reasoning.append("Price above SMA20 - Short-term bullish")
            elif current_price < sma_50:
                signals['trend_signal'] = -0.6
                reasoning.append("Price below SMA50 - Bearish trend")
            else:
                signals['trend_signal'] = 0
        
        # RSI signal
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                signals['momentum_signal'] = 0.7
                reasoning.append(f"RSI oversold at {current_rsi:.1f} - Potential bounce")
            elif current_rsi > 70:
                signals['momentum_signal'] = -0.7
                reasoning.append(f"RSI overbought at {current_rsi:.1f} - Potential pullback")
            else:
                signals['momentum_signal'] = 0
        
        # Volume signal
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        recent_volume = data['Volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 2:
            signals['volume_signal'] = 0.5
            reasoning.append(f"High volume ({volume_ratio:.1f}x average) - Increased interest")
        elif volume_ratio < 0.5:
            signals['volume_signal'] = -0.3
            reasoning.append("Low volume - Lack of conviction")
        else:
            signals['volume_signal'] = 0
        
        # Price momentum
        price_change_1d = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        price_change_5d = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] if len(data) >= 6 else 0
        
        if price_change_1d > 0.03:
            signals['price_momentum'] = 0.6
            reasoning.append(f"Strong daily gain of {price_change_1d*100:.1f}%")
        elif price_change_5d > 0.1:
            signals['price_momentum'] = 0.4
            reasoning.append(f"Strong 5-day gain of {price_change_5d*100:.1f}%")
        elif price_change_1d < -0.03:
            signals['price_momentum'] = -0.6
            reasoning.append(f"Sharp daily decline of {price_change_1d*100:.1f}%")
        else:
            signals['price_momentum'] = 0
        
        # Catalyst boost
        if catalyst_data:
            catalyst_impact = catalyst_data.get('impact_score', 0) / 100  # Normalize to -1 to 1
            signals['catalyst_signal'] = catalyst_impact
            if catalyst_impact > 0.5:
                reasoning.append(f"Strong positive catalyst detected: {catalyst_data.get('category', 'Unknown')}")
            elif catalyst_impact < -0.5:
                reasoning.append(f"Negative catalyst detected: {catalyst_data.get('category', 'Unknown')}")
        
        # Calculate overall signal
        signal_weights = {
            'trend_signal': 0.3,
            'momentum_signal': 0.25,
            'volume_signal': 0.15,
            'price_momentum': 0.2,
            'catalyst_signal': 0.1
        }
        
        overall_signal = sum(signals.get(key, 0) * weight for key, weight in signal_weights.items())
        
        return {
            'overall_signal': overall_signal,
            'signal_breakdown': signals,
            'reasoning': reasoning
        }
        
    except Exception as e:
        logger.error(f"Error generating trading signals for {ticker}: {e}")
        return {
            'overall_signal': 0,
            'signal_breakdown': {},
            'reasoning': ['Error analyzing stock data']
        }