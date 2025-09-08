"""
Comprehensive Stock Screener with Trading Strategy Presets
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Trading Strategy Presets based on the framework
TRADING_STRATEGY_PRESETS = {
    "Day Trading - Scalping": {
        "description": "High-frequency intraday trades using tick charts and Level II data",
        "timeframe": "1m-5m",
        "technical": {
            "rsi_min": None, "rsi_max": None,
            "volume_min": 1000000,  # High volume required
            "price_change_min": -5, "price_change_max": 5,
            "volatility_min": 2.0,  # High volatility
            "moving_averages": {"use_vwap": True, "use_sma_20": True}
        },
        "fundamental": {
            "market_cap_min": 1000000000,  # Large cap for liquidity
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": None,
            "news_impact_min": None
        },
        "signal_weights": {
            "technical": 0.80, "sentiment": 0.15, "catalyst": 0.05, "fundamental": 0.00
        }
    },
    
    "Day Trading - Momentum": {
        "description": "Momentum-based day trading using RSI, MACD, and news catalysts",
        "timeframe": "5m-15m",
        "technical": {
            "rsi_min": 70, "rsi_max": None,  # Overbought momentum
            "volume_min": 500000,
            "price_change_min": 2, "price_change_max": None,
            "volatility_min": 1.5,
            "moving_averages": {"use_ema_12": True, "use_ema_26": True}
        },
        "fundamental": {
            "market_cap_min": 500000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": 0.6,  # Positive sentiment
            "news_impact_min": 60
        },
        "signal_weights": {
            "technical": 0.60, "sentiment": 0.25, "catalyst": 0.15, "fundamental": 0.00
        }
    },
    
    "Day Trading - Reversal": {
        "description": "Reversal patterns using candlestick analysis and volume divergence",
        "timeframe": "5m-30m", 
        "technical": {
            "rsi_min": None, "rsi_max": 30,  # Oversold for reversal
            "volume_min": 300000,
            "price_change_min": None, "price_change_max": -2,  # Recent decline
            "volatility_min": 1.0,
            "moving_averages": {"use_sma_20": True, "use_bollinger": True}
        },
        "fundamental": {
            "market_cap_min": 300000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": None, "sentiment_score_max": 0.4,  # Negative sentiment for reversal
            "news_impact_min": None
        },
        "signal_weights": {
            "technical": 0.70, "sentiment": 0.20, "catalyst": 0.10, "fundamental": 0.00
        }
    },
    
    "Swing Trading - Trend Following": {
        "description": "Multi-day trend following using moving averages and MACD",
        "timeframe": "1h-4h",
        "technical": {
            "rsi_min": 40, "rsi_max": 80,
            "volume_min": 100000,
            "price_change_min": None, "price_change_max": None,
            "volatility_min": 0.5,
            "moving_averages": {"use_sma_50": True, "use_sma_200": True, "golden_cross": True}
        },
        "fundamental": {
            "market_cap_min": 100000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": 0.5,
            "news_impact_min": 40
        },
        "signal_weights": {
            "technical": 0.45, "sentiment": 0.25, "catalyst": 0.20, "fundamental": 0.10
        }
    },
    
    "Swing Trading - Mean Reversion": {
        "description": "Mean reversion using Bollinger Bands and RSI oversold/overbought",
        "timeframe": "4h-1d",
        "technical": {
            "rsi_min": None, "rsi_max": 35,  # Oversold
            "volume_min": 50000,
            "price_change_min": None, "price_change_max": -5,
            "volatility_min": 0.8,
            "moving_averages": {"use_bollinger": True, "use_sma_20": True}
        },
        "fundamental": {
            "market_cap_min": 50000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": None, "sentiment_score_max": 0.3,
            "news_impact_min": None
        },
        "signal_weights": {
            "technical": 0.50, "sentiment": 0.20, "catalyst": 0.15, "fundamental": 0.15
        }
    },
    
    "Position Trading - Value Investing": {
        "description": "Long-term value investing using P/E, P/B ratios and DCF models",
        "timeframe": "1d-1w",
        "technical": {
            "rsi_min": None, "rsi_max": None,
            "volume_min": None,
            "price_change_min": None, "price_change_max": None,
            "volatility_min": None,
            "moving_averages": {"use_sma_200": True}
        },
        "fundamental": {
            "market_cap_min": 1000000000,  # Large cap
            "market_cap_max": None,
            "pe_ratio_min": None, "pe_ratio_max": 20,  # Value stocks
            "pb_ratio_min": None, "pb_ratio_max": 3,
            "debt_to_equity_max": 0.5,
            "roe_min": 10
        },
        "sentiment": {
            "sentiment_score_min": None,
            "news_impact_min": None
        },
        "signal_weights": {
            "technical": 0.25, "sentiment": 0.15, "catalyst": 0.20, "fundamental": 0.40
        }
    },
    
    "Position Trading - Growth Investing": {
        "description": "Growth investing focusing on revenue growth and market expansion",
        "timeframe": "1d-1w",
        "technical": {
            "rsi_min": 40, "rsi_max": None,
            "volume_min": None,
            "price_change_min": None, "price_change_max": None,
            "volatility_min": None,
            "moving_averages": {"use_sma_50": True, "use_sma_200": True}
        },
        "fundamental": {
            "market_cap_min": 500000000,
            "market_cap_max": None,
            "revenue_growth_min": 15,  # Strong growth
            "earnings_growth_min": 20,
            "pe_ratio_min": None, "pe_ratio_max": 50  # Growth premium acceptable
        },
        "sentiment": {
            "sentiment_score_min": 0.6,  # Positive outlook
            "news_impact_min": 50
        },
        "signal_weights": {
            "technical": 0.25, "sentiment": 0.15, "catalyst": 0.20, "fundamental": 0.40
        }
    },
    
    "Sentiment Trading - Contrarian": {
        "description": "Contrarian strategy using extreme sentiment and Put/Call ratios",
        "timeframe": "1h-1d",
        "technical": {
            "rsi_min": None, "rsi_max": 25,  # Extremely oversold
            "volume_min": 100000,
            "price_change_min": None, "price_change_max": -10,  # Major decline
            "volatility_min": 2.0,
            "moving_averages": {"use_sma_20": True}
        },
        "fundamental": {
            "market_cap_min": 100000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": None, "sentiment_score_max": 0.2,  # Very negative sentiment
            "news_impact_min": 70  # High impact negative news
        },
        "signal_weights": {
            "technical": 0.30, "sentiment": 0.50, "catalyst": 0.20, "fundamental": 0.00
        }
    },
    
    "Sentiment Trading - Momentum": {
        "description": "Follow crowd momentum using social media buzz and Google Trends",
        "timeframe": "15m-4h",
        "technical": {
            "rsi_min": 60, "rsi_max": None,  # Strong momentum
            "volume_min": 200000,
            "price_change_min": 5, "price_change_max": None,  # Strong upward move
            "volatility_min": 1.5,
            "moving_averages": {"use_ema_12": True, "use_ema_26": True}
        },
        "fundamental": {
            "market_cap_min": 50000000,
            "market_cap_max": None
        },
        "sentiment": {
            "sentiment_score_min": 0.8,  # Very positive sentiment
            "news_impact_min": 80,  # High impact positive news
            "social_buzz_min": 70  # High social media activity
        },
        "signal_weights": {
            "technical": 0.25, "sentiment": 0.60, "catalyst": 0.15, "fundamental": 0.00
        }
    }
}

def render_stock_screener_page(app_components: Dict[str, Any]):
    """Render comprehensive stock screener with strategy presets"""
    st.title("🔍 Advanced Stock Screener")
    st.markdown("Find stocks matching your trading strategy with advanced technical, fundamental, and sentiment filters.")
    
    # Strategy Preset Selection
    st.markdown("---")
    st.subheader("🎯 Trading Strategy Presets")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        preset_names = ["Custom (Manual Settings)"] + list(TRADING_STRATEGY_PRESETS.keys())
        selected_preset = st.selectbox(
            "Choose a Trading Strategy",
            preset_names,
            help="Select a preset to automatically configure all screening parameters"
        )
    
    with col2:
        if selected_preset != "Custom (Manual Settings)":
            preset_info = TRADING_STRATEGY_PRESETS[selected_preset]
            st.info(f"**Timeframe:** {preset_info['timeframe']}\n\n{preset_info['description']}")
    
    # Initialize session state for screener settings
    if 'screener_settings' not in st.session_state:
        st.session_state.screener_settings = {}
    
    # Apply preset if selected
    if selected_preset != "Custom (Manual Settings)" and st.button("🎯 Apply Preset"):
        st.session_state.screener_settings = TRADING_STRATEGY_PRESETS[selected_preset].copy()
        st.success(f"✅ Applied {selected_preset} preset!")
        st.rerun()
    
    # Create tabs for different filter categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Technical", "📊 Fundamental", "💭 Sentiment", "⚙️ Advanced", "🏆 Results"
    ])
    
    with tab1:
        render_technical_filters()
    
    with tab2:
        render_fundamental_filters()
    
    with tab3:
        render_sentiment_filters()
    
    with tab4:
        render_advanced_filters()
    
    with tab5:
        render_screening_results(app_components)

def render_technical_filters():
    """Render technical analysis filters"""
    st.header("📈 Technical Analysis Filters")
    
    current_settings = st.session_state.screener_settings.get('technical', {})
    
    # Price and Volume
    st.subheader("💰 Price & Volume")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Price Change (%)**")
        price_change_min = st.number_input(
            "Minimum %", 
            value=current_settings.get('price_change_min', -50),
            min_value=-50, max_value=50,
            help="Minimum price change percentage (negative for decline)"
        )
        price_change_max = st.number_input(
            "Maximum %", 
            value=current_settings.get('price_change_max', 50),
            min_value=-50, max_value=50,
            help="Maximum price change percentage"
        )
    
    with col2:
        st.markdown("**Volume**")
        volume_min = st.number_input(
            "Minimum Volume", 
            value=current_settings.get('volume_min', 100000),
            min_value=0, max_value=10000000, step=50000,
            help="Minimum daily trading volume"
        )
        
        volatility_min = st.number_input(
            "Minimum Volatility (%)", 
            value=current_settings.get('volatility_min', 1.0),
            min_value=0.0, max_value=10.0, step=0.1,
            help="Minimum daily volatility percentage"
        )
    
    # RSI Settings
    st.subheader("📊 RSI (Relative Strength Index)")
    col1, col2 = st.columns(2)
    
    with col1:
        rsi_min = st.number_input(
            "RSI Minimum", 
            value=current_settings.get('rsi_min', 30),
            min_value=0, max_value=100,
            help="Minimum RSI value (30 = oversold)"
        )
    
    with col2:
        rsi_max = st.number_input(
            "RSI Maximum", 
            value=current_settings.get('rsi_max', 70),
            min_value=0, max_value=100,
            help="Maximum RSI value (70 = overbought)"
        )
    
    # Moving Averages
    st.subheader("📈 Moving Averages & Patterns")
    ma_settings = current_settings.get('moving_averages', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_sma_20 = st.checkbox("SMA 20", value=ma_settings.get('use_sma_20', False))
        use_sma_50 = st.checkbox("SMA 50", value=ma_settings.get('use_sma_50', False))
        use_sma_200 = st.checkbox("SMA 200", value=ma_settings.get('use_sma_200', False))
    
    with col2:
        use_ema_12 = st.checkbox("EMA 12", value=ma_settings.get('use_ema_12', False))
        use_ema_26 = st.checkbox("EMA 26", value=ma_settings.get('use_ema_26', False))
        use_vwap = st.checkbox("VWAP", value=ma_settings.get('use_vwap', False))
    
    with col3:
        use_bollinger = st.checkbox("Bollinger Bands", value=ma_settings.get('use_bollinger', False))
        golden_cross = st.checkbox("Golden Cross (50>200)", value=ma_settings.get('golden_cross', False))
        death_cross = st.checkbox("Death Cross (50<200)", value=ma_settings.get('death_cross', False))
    
    # Save technical settings to session state
    if 'screener_settings' not in st.session_state:
        st.session_state.screener_settings = {}
    
    st.session_state.screener_settings['technical'] = {
        'price_change_min': price_change_min if price_change_min != -50 else None,
        'price_change_max': price_change_max if price_change_max != 50 else None,
        'volume_min': volume_min if volume_min > 0 else None,
        'volatility_min': volatility_min if volatility_min > 0 else None,
        'rsi_min': rsi_min if rsi_min > 0 else None,
        'rsi_max': rsi_max if rsi_max < 100 else None,
        'moving_averages': {
            'use_sma_20': use_sma_20,
            'use_sma_50': use_sma_50,
            'use_sma_200': use_sma_200,
            'use_ema_12': use_ema_12,
            'use_ema_26': use_ema_26,
            'use_vwap': use_vwap,
            'use_bollinger': use_bollinger,
            'golden_cross': golden_cross,
            'death_cross': death_cross
        }
    }

def render_fundamental_filters():
    """Render fundamental analysis filters"""
    st.header("📊 Fundamental Analysis Filters")
    
    current_settings = st.session_state.screener_settings.get('fundamental', {})
    
    # Market Cap
    st.subheader("🏢 Market Capitalization")
    col1, col2 = st.columns(2)
    
    with col1:
        market_cap_options = {
            "Any": None,
            "Micro Cap (<$300M)": (0, 300000000),
            "Small Cap ($300M-$2B)": (300000000, 2000000000),
            "Mid Cap ($2B-$10B)": (2000000000, 10000000000),
            "Large Cap ($10B-$200B)": (10000000000, 200000000000),
            "Mega Cap (>$200B)": (200000000000, None)
        }
        
        market_cap_selection = st.selectbox(
            "Market Cap Range",
            list(market_cap_options.keys()),
            help="Filter by company size"
        )
        
        if market_cap_selection != "Any":
            cap_range = market_cap_options[market_cap_selection]
            market_cap_min = cap_range[0]
            market_cap_max = cap_range[1]
        else:
            market_cap_min = None
            market_cap_max = None
    
    with col2:
        # Custom market cap
        st.markdown("**Custom Range (Optional)**")
        custom_cap_min = st.number_input(
            "Min Market Cap ($M)", 
            value=current_settings.get('market_cap_min', 0) // 1000000 if current_settings.get('market_cap_min') else 0,
            min_value=0, max_value=1000000, step=100
        ) * 1000000
        
        custom_cap_max = st.number_input(
            "Max Market Cap ($M)", 
            value=current_settings.get('market_cap_max', 0) // 1000000 if current_settings.get('market_cap_max') else 0,
            min_value=0, max_value=1000000, step=100
        ) * 1000000
    
    # Use custom values if provided
    if custom_cap_min > 0:
        market_cap_min = custom_cap_min
    if custom_cap_max > 0:
        market_cap_max = custom_cap_max
    
    # Valuation Ratios
    st.subheader("💹 Valuation Ratios")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**P/E Ratio**")
        pe_ratio_min = st.number_input(
            "P/E Min", 
            value=current_settings.get('pe_ratio_min', 0),
            min_value=0, max_value=200, step=1,
            help="Minimum Price-to-Earnings ratio"
        )
        pe_ratio_max = st.number_input(
            "P/E Max", 
            value=current_settings.get('pe_ratio_max', 50),
            min_value=0, max_value=200, step=1,
            help="Maximum Price-to-Earnings ratio"
        )
    
    with col2:
        st.markdown("**P/B Ratio**")
        pb_ratio_min = st.number_input(
            "P/B Min", 
            value=current_settings.get('pb_ratio_min', 0.0),
            min_value=0.0, max_value=20.0, step=0.1,
            help="Minimum Price-to-Book ratio"
        )
        pb_ratio_max = st.number_input(
            "P/B Max", 
            value=current_settings.get('pb_ratio_max', 5.0),
            min_value=0.0, max_value=20.0, step=0.1,
            help="Maximum Price-to-Book ratio"
        )
    
    # Financial Health
    st.subheader("💪 Financial Health")
    col1, col2 = st.columns(2)
    
    with col1:
        debt_to_equity_max = st.number_input(
            "Max Debt/Equity", 
            value=current_settings.get('debt_to_equity_max', 1.0),
            min_value=0.0, max_value=5.0, step=0.1,
            help="Maximum debt-to-equity ratio"
        )
        
        roe_min = st.number_input(
            "Min ROE (%)", 
            value=current_settings.get('roe_min', 10),
            min_value=0, max_value=100, step=1,
            help="Minimum Return on Equity percentage"
        )
    
    with col2:
        revenue_growth_min = st.number_input(
            "Min Revenue Growth (%)", 
            value=current_settings.get('revenue_growth_min', 0),
            min_value=-50, max_value=200, step=1,
            help="Minimum revenue growth percentage"
        )
        
        earnings_growth_min = st.number_input(
            "Min Earnings Growth (%)", 
            value=current_settings.get('earnings_growth_min', 0),
            min_value=-100, max_value=500, step=5,
            help="Minimum earnings growth percentage"
        )
    
    # Save fundamental settings
    st.session_state.screener_settings['fundamental'] = {
        'market_cap_min': market_cap_min,
        'market_cap_max': market_cap_max,
        'pe_ratio_min': pe_ratio_min if pe_ratio_min > 0 else None,
        'pe_ratio_max': pe_ratio_max if pe_ratio_max < 200 else None,
        'pb_ratio_min': pb_ratio_min if pb_ratio_min > 0 else None,
        'pb_ratio_max': pb_ratio_max if pb_ratio_max < 20 else None,
        'debt_to_equity_max': debt_to_equity_max if debt_to_equity_max < 5.0 else None,
        'roe_min': roe_min if roe_min > 0 else None,
        'revenue_growth_min': revenue_growth_min if revenue_growth_min != 0 else None,
        'earnings_growth_min': earnings_growth_min if earnings_growth_min != 0 else None
    }

def render_sentiment_filters():
    """Render sentiment analysis filters"""
    st.header("💭 Sentiment Analysis Filters")
    
    current_settings = st.session_state.screener_settings.get('sentiment', {})
    
    # Sentiment Scores
    st.subheader("📊 Sentiment Scores")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_score_min = st.slider(
            "Minimum Sentiment Score",
            min_value=0.0, max_value=1.0, 
            value=current_settings.get('sentiment_score_min', 0.5),
            step=0.05,
            help="0.0 = Very Negative, 1.0 = Very Positive"
        )
    
    with col2:
        sentiment_score_max = st.slider(
            "Maximum Sentiment Score",
            min_value=0.0, max_value=1.0, 
            value=current_settings.get('sentiment_score_max', 1.0),
            step=0.05,
            help="Filter for contrarian plays (low sentiment)"
        )
    
    # News Impact
    st.subheader("📰 News & Catalyst Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        news_impact_min = st.slider(
            "Minimum News Impact",
            min_value=0, max_value=100,
            value=current_settings.get('news_impact_min', 50),
            help="Minimum catalyst impact score"
        )
    
    with col2:
        social_buzz_min = st.slider(
            "Minimum Social Buzz",
            min_value=0, max_value=100,
            value=current_settings.get('social_buzz_min', 30),
            help="Minimum social media activity level"
        )
    
    # Sentiment Sources
    st.subheader("📱 Sentiment Sources")
    sentiment_sources = st.multiselect(
        "Include Sentiment From",
        ["News Articles", "Twitter", "Reddit", "Financial Reports", "Analyst Reports"],
        default=current_settings.get('sentiment_sources', ["News Articles", "Twitter"]),
        help="Select which sources to include in sentiment analysis"
    )
    
    # Sentiment Timeframe
    sentiment_timeframe = st.selectbox(
        "Sentiment Timeframe",
        ["1 Hour", "4 Hours", "1 Day", "3 Days", "1 Week"],
        index=2,  # Default to 1 Day
        help="Time period for sentiment analysis"
    )
    
    # Save sentiment settings
    st.session_state.screener_settings['sentiment'] = {
        'sentiment_score_min': sentiment_score_min if sentiment_score_min > 0.0 else None,
        'sentiment_score_max': sentiment_score_max if sentiment_score_max < 1.0 else None,
        'news_impact_min': news_impact_min if news_impact_min > 0 else None,
        'social_buzz_min': social_buzz_min if social_buzz_min > 0 else None,
        'sentiment_sources': sentiment_sources,
        'sentiment_timeframe': sentiment_timeframe
    }

def render_advanced_filters():
    """Render advanced screening filters"""
    st.header("⚙️ Advanced Filters")
    
    current_settings = st.session_state.screener_settings.get('advanced', {})
    
    # Sector and Industry
    st.subheader("🏭 Sector & Industry")
    
    sectors = [
        "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Utilities", "Real Estate",
        "Communication Services", "Materials", "Industrials"
    ]
    
    selected_sectors = st.multiselect(
        "Sectors to Include",
        sectors,
        default=current_settings.get('sectors', []),
        help="Leave empty to include all sectors"
    )
    
    # Exchange and Region
    st.subheader("🌍 Exchange & Region")
    col1, col2 = st.columns(2)
    
    with col1:
        exchanges = st.multiselect(
            "Exchanges",
            ["NYSE", "NASDAQ", "AMEX"],
            default=current_settings.get('exchanges', ["NYSE", "NASDAQ"]),
            help="Stock exchanges to include"
        )
    
    with col2:
        regions = st.multiselect(
            "Regions",
            ["North America", "Europe", "Asia", "Other"],
            default=current_settings.get('regions', ["North America"]),
            help="Geographic regions to include"
        )
    
    # Risk Metrics
    st.subheader("⚠️ Risk Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        beta_min = st.number_input(
            "Minimum Beta",
            value=current_settings.get('beta_min', 0.0),
            min_value=0.0, max_value=5.0, step=0.1,
            help="Stock volatility relative to market (1.0 = market average)"
        )
        
        beta_max = st.number_input(
            "Maximum Beta",
            value=current_settings.get('beta_max', 3.0),
            min_value=0.0, max_value=5.0, step=0.1,
            help="Maximum beta for risk control"
        )
    
    with col2:
        sharpe_ratio_min = st.number_input(
            "Minimum Sharpe Ratio",
            value=current_settings.get('sharpe_ratio_min', 0.0),
            min_value=-2.0, max_value=5.0, step=0.1,
            help="Risk-adjusted return metric"
        )
    
    # Results and Performance
    st.subheader("📈 Results Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        max_results = st.number_input(
            "Maximum Results",
            value=current_settings.get('max_results', 50),
            min_value=10, max_value=500, step=10,
            help="Maximum number of stocks to return"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort Results By",
            ["Signal Strength", "Market Cap", "Volume", "Price Change", "Sentiment Score"],
            index=0,
            help="How to sort the screening results"
        )
    
    # Signal Weight Customization
    st.subheader("⚖️ Signal Weight Customization")
    st.markdown("Customize how different analyses are weighted in the final signal:")
    
    current_weights = current_settings.get('signal_weights', {
        'technical': 0.40, 'fundamental': 0.30, 'sentiment': 0.20, 'catalyst': 0.10
    })
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        technical_weight = st.slider(
            "Technical Analysis",
            min_value=0.0, max_value=1.0,
            value=current_weights.get('technical', 0.40),
            step=0.05
        )
    
    with col2:
        fundamental_weight = st.slider(
            "Fundamental Analysis",
            min_value=0.0, max_value=1.0,
            value=current_weights.get('fundamental', 0.30),
            step=0.05
        )
    
    with col3:
        sentiment_weight = st.slider(
            "Sentiment Analysis",
            min_value=0.0, max_value=1.0,
            value=current_weights.get('sentiment', 0.20),
            step=0.05
        )
    
    with col4:
        catalyst_weight = st.slider(
            "Catalyst Impact",
            min_value=0.0, max_value=1.0,
            value=current_weights.get('catalyst', 0.10),
            step=0.05
        )
    
    # Normalize weights to sum to 1.0
    total_weight = technical_weight + fundamental_weight + sentiment_weight + catalyst_weight
    if total_weight > 0:
        technical_weight /= total_weight
        fundamental_weight /= total_weight
        sentiment_weight /= total_weight
        catalyst_weight /= total_weight
        
        st.info(f"**Normalized Weights:** Technical: {technical_weight:.1%}, Fundamental: {fundamental_weight:.1%}, Sentiment: {sentiment_weight:.1%}, Catalyst: {catalyst_weight:.1%}")
    
    # Save advanced settings
    st.session_state.screener_settings['advanced'] = {
        'sectors': selected_sectors,
        'exchanges': exchanges,
        'regions': regions,
        'beta_min': beta_min if beta_min > 0.0 else None,
        'beta_max': beta_max if beta_max < 5.0 else None,
        'sharpe_ratio_min': sharpe_ratio_min if sharpe_ratio_min > -2.0 else None,
        'max_results': max_results,
        'sort_by': sort_by,
        'signal_weights': {
            'technical': technical_weight,
            'fundamental': fundamental_weight,
            'sentiment': sentiment_weight,
            'catalyst': catalyst_weight
        }
    }

def render_screening_results(app_components: Dict[str, Any]):
    """Render screening results"""
    st.header("🏆 Screening Results")
    
    if st.button("🔍 Run Screen", type="primary", use_container_width=True):
        with st.spinner("🔍 Scanning stocks based on your criteria..."):
            try:
                # Get screener settings
                settings = st.session_state.get('screener_settings', {})
                
                # Run the stock screening
                results = run_stock_screen(app_components, settings)
                
                if results and len(results) > 0:
                    st.success(f"✅ Found {len(results)} stocks matching your criteria!")
                    
                    # Create results DataFrame
                    df = pd.DataFrame(results)
                    
                    # Display results table
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Signal Strength": st.column_config.ProgressColumn(
                                "Signal Strength",
                                help="Overall signal strength (0-100)",
                                min_value=0,
                                max_value=100,
                            ),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                            "Market Cap": st.column_config.NumberColumn("Market Cap", format="$%.0f M"),
                            "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                            "Sector": st.column_config.TextColumn("Sector"),
                        }
                    )
                    
                    # Display summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_signal = df['Signal Strength'].mean()
                        st.metric("Average Signal", f"{avg_signal:.1f}")
                    
                    with col2:
                        avg_change = df['Change %'].mean()
                        st.metric("Average Change", f"{avg_change:.2f}%")
                    
                    with col3:
                        total_volume = df['Volume'].sum()
                        st.metric("Total Volume", f"{total_volume:,.0f}")
                    
                    with col4:
                        unique_sectors = int(df['Sector'].nunique())
                        st.metric("Sectors Found", unique_sectors)
                    
                else:
                    st.warning("⚠️ No stocks found matching your criteria. Try adjusting your filters.")
                    
            except Exception as e:
                st.error(f"❌ Error running screen: {str(e)}")
                logger.error(f"Screening error: {e}")
    
    else:
        st.info("👆 Configure your filters above and click 'Run Screen' to find matching stocks.")

def run_stock_screen(app_components: Dict[str, Any], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run the actual stock screening logic"""
    try:
        # Get the signal generator and database manager from app components
        signal_generator = app_components.get('signal_generator')
        db_manager = app_components.get('db_manager')
        
        # Generate real stock universe from live market data
        stock_universe = generate_real_stock_universe()
        
        # Apply screening filters
        filtered_stocks = apply_screening_filters(stock_universe, settings)
        
        # Calculate ensemble signals for each filtered stock
        results = []
        for stock in filtered_stocks:
            try:
                # Calculate composite signal using existing ensemble model
                if signal_generator:
                    # Get strategy weights from settings
                    weights = settings.get('advanced', {}).get('signal_weights', {
                        'technical': 0.40, 'fundamental': 0.30, 'sentiment': 0.20, 'catalyst': 0.10
                    })
                    
                    # Calculate individual signals (placeholder - real implementation would use actual data)
                    technical_signal = calculate_technical_signal(stock)
                    fundamental_signal = calculate_fundamental_signal(stock)
                    sentiment_signal = calculate_sentiment_signal(stock)
                    catalyst_signal = calculate_catalyst_signal(stock, db_manager)
                    
                    # Calculate weighted composite signal
                    composite_signal = (
                        technical_signal * weights.get('technical', 0.4) +
                        fundamental_signal * weights.get('fundamental', 0.3) +
                        sentiment_signal * weights.get('sentiment', 0.2) +
                        catalyst_signal * weights.get('catalyst', 0.1)
                    ) * 100  # Scale to 0-100
                    
                    stock['Signal Strength'] = min(100, max(0, round(composite_signal)))
                
                results.append(stock)
                
            except Exception as e:
                logger.error(f"Error calculating signal for {stock.get('Ticker', 'Unknown')}: {e}")
                continue
        
        # Sort by signal strength
        sort_by = settings.get('advanced', {}).get('sort_by', 'Signal Strength')
        if sort_by == 'Signal Strength':
            results.sort(key=lambda x: x.get('Signal Strength', 0), reverse=True)
        elif sort_by == 'Market Cap':
            results.sort(key=lambda x: x.get('Market Cap', 0), reverse=True)
        elif sort_by == 'Volume':
            results.sort(key=lambda x: x.get('Volume', 0), reverse=True)
        elif sort_by == 'Price Change':
            results.sort(key=lambda x: x.get('Change %', 0), reverse=True)
        elif sort_by == 'Sentiment Score':
            results.sort(key=lambda x: x.get('Sentiment', 0), reverse=True)
        
        # Limit results
        max_results = settings.get('advanced', {}).get('max_results', 50)
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"Stock screening failed: {e}")
        return []

def generate_real_stock_universe() -> List[Dict[str, Any]]:
    """Generate real stock universe from live market data sources"""
    import yfinance as yf
    import requests
    from datetime import datetime, timedelta
    import numpy as np
    
    logger.info("Fetching real stock data from live sources...")
    
    # Major stock tickers for screening - real actively traded stocks
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 
        'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'NFLX', 'HD', 'BAC', 'ADBE',
        'CRM', 'KO', 'PFE', 'ABBV', 'TMO', 'ACN', 'VZ', 'NKE', 'MRK', 'PYPL',
        'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'ORCL', 'IBM', 'CSCO', 'CVX', 'XOM'
    ]
    
    stocks = []
    
    try:
        # Fetch data for each ticker using yfinance (free and reliable)
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="30d", interval="1d")
                
                if hist.empty or not info:
                    logger.warning(f"No data available for {ticker}")
                    continue
                
                # Extract real market data
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                volume = int(hist['Volume'].iloc[-1])
                avg_volume = int(hist['Volume'].mean())
                
                # Calculate real technical indicators
                prices = hist['Close']
                high_prices = hist['High']
                low_prices = hist['Low']
                
                # Real RSI calculation
                rsi = calculate_real_rsi(prices) if len(prices) >= 14 else 50
                
                # Real price change
                price_change = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                # Real volatility (30-day)
                returns = prices.pct_change().dropna()
                volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else 0
                
                # Extract fundamental data from yfinance info
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                pb_ratio = info.get('priceToBook', 0)
                roe = info.get('returnOnEquity', 0)
                debt_to_equity = info.get('debtToEquity', 0)
                revenue_growth = info.get('revenueGrowth', 0)
                earnings_growth = info.get('earningsGrowth', 0)
                beta = info.get('beta', 1.0)
                sector = info.get('sector', 'Unknown')
                
                # Clean and validate data
                pe_ratio = pe_ratio if pe_ratio and pe_ratio > 0 and pe_ratio < 1000 else 0
                pb_ratio = pb_ratio if pb_ratio and pb_ratio > 0 and pb_ratio < 100 else 0
                roe = (roe * 100) if roe and -100 <= roe <= 100 else 0
                debt_to_equity = debt_to_equity if debt_to_equity and debt_to_equity >= 0 and debt_to_equity < 10 else 0
                revenue_growth = (revenue_growth * 100) if revenue_growth and -100 <= revenue_growth <= 200 else 0
                earnings_growth = (earnings_growth * 100) if earnings_growth and -200 <= earnings_growth <= 500 else 0
                beta = beta if beta and 0 <= beta <= 5 else 1.0
                
                # Calculate sentiment score based on recent price action and volume
                sentiment = calculate_market_sentiment(hist, info)
                
                stock_data = {
                    "Ticker": ticker,
                    "Sector": sector,
                    "Price": round(current_price, 2),
                    "Change %": round(price_change, 2),
                    "Volume": volume,
                    "Avg Volume": avg_volume,
                    "Market Cap": market_cap,
                    "RSI": round(rsi, 1),
                    "P/E": round(pe_ratio, 1) if pe_ratio else 0,
                    "P/B": round(pb_ratio, 1) if pb_ratio else 0,
                    "ROE": round(roe, 1),
                    "Debt/Equity": round(debt_to_equity, 2),
                    "Revenue Growth": round(revenue_growth, 1),
                    "Earnings Growth": round(earnings_growth, 1),
                    "Beta": round(beta, 2),
                    "Sentiment": round(sentiment, 2),
                    "Volatility": round(volatility, 1),
                    "52W High": info.get('fiftyTwoWeekHigh', current_price),
                    "52W Low": info.get('fiftyTwoWeekLow', current_price),
                    "Div Yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                stocks.append(stock_data)
                logger.info(f"Successfully fetched data for {ticker}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                continue
        
        logger.info(f"Successfully fetched real data for {len(stocks)} stocks")
        return stocks
        
    except Exception as e:
        logger.error(f"Error fetching real stock universe: {e}")
        # Return empty list to force error handling rather than fallback to fake data
        return []

def calculate_real_rsi(prices, window=14):
    """Calculate real RSI from price data"""
    if len(prices) < window + 1:
        return 50
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

def calculate_market_sentiment(hist_data, info):
    """Calculate sentiment score based on real market data"""
    try:
        prices = hist_data['Close']
        volumes = hist_data['Volume']
        
        if len(prices) < 10:
            return 0.5
        
        # Price momentum (last 10 days vs previous 10 days)
        recent_avg = prices.tail(10).mean()
        previous_avg = prices.iloc[-20:-10].mean() if len(prices) >= 20 else recent_avg
        price_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
        
        # Volume momentum
        recent_vol = volumes.tail(5).mean()
        avg_vol = volumes.mean()
        volume_momentum = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
        
        # Combine factors
        sentiment = 0.5 + (price_momentum * 2) + (volume_momentum * 0.5)
        
        # Bound between 0.1 and 0.9
        return max(0.1, min(0.9, sentiment))
        
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {e}")
        return 0.5

def apply_screening_filters(stocks: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply all screening filters to the stock universe"""
    filtered_stocks = []
    
    technical_settings = settings.get('technical', {})
    fundamental_settings = settings.get('fundamental', {})
    sentiment_settings = settings.get('sentiment', {})
    advanced_settings = settings.get('advanced', {})
    
    for stock in stocks:
        # Technical filters
        if technical_settings.get('rsi_min') and stock['RSI'] < technical_settings['rsi_min']:
            continue
        if technical_settings.get('rsi_max') and stock['RSI'] > technical_settings['rsi_max']:
            continue
        if technical_settings.get('volume_min') and stock['Volume'] < technical_settings['volume_min']:
            continue
        if technical_settings.get('volatility_min') and stock['Volatility'] < technical_settings['volatility_min']:
            continue
        if technical_settings.get('price_change_min') and stock['Change %'] < technical_settings['price_change_min']:
            continue
        if technical_settings.get('price_change_max') and stock['Change %'] > technical_settings['price_change_max']:
            continue
        
        # Fundamental filters
        if fundamental_settings.get('market_cap_min') and stock['Market Cap'] * 1000000 < fundamental_settings['market_cap_min']:
            continue
        if fundamental_settings.get('market_cap_max') and stock['Market Cap'] * 1000000 > fundamental_settings['market_cap_max']:
            continue
        if fundamental_settings.get('pe_ratio_min') and stock['P/E'] < fundamental_settings['pe_ratio_min']:
            continue
        if fundamental_settings.get('pe_ratio_max') and stock['P/E'] > fundamental_settings['pe_ratio_max']:
            continue
        if fundamental_settings.get('pb_ratio_min') and stock['P/B'] < fundamental_settings['pb_ratio_min']:
            continue
        if fundamental_settings.get('pb_ratio_max') and stock['P/B'] > fundamental_settings['pb_ratio_max']:
            continue
        if fundamental_settings.get('roe_min') and stock['ROE'] < fundamental_settings['roe_min']:
            continue
        if fundamental_settings.get('debt_to_equity_max') and stock['Debt/Equity'] > fundamental_settings['debt_to_equity_max']:
            continue
        if fundamental_settings.get('revenue_growth_min') and stock['Revenue Growth'] < fundamental_settings['revenue_growth_min']:
            continue
        if fundamental_settings.get('earnings_growth_min') and stock['Earnings Growth'] < fundamental_settings['earnings_growth_min']:
            continue
        
        # Sentiment filters
        if sentiment_settings.get('sentiment_score_min') and stock['Sentiment'] < sentiment_settings['sentiment_score_min']:
            continue
        if sentiment_settings.get('sentiment_score_max') and stock['Sentiment'] > sentiment_settings['sentiment_score_max']:
            continue
        
        # Advanced filters
        if advanced_settings.get('sectors') and stock['Sector'] not in advanced_settings['sectors']:
            continue
        if advanced_settings.get('beta_min') and stock['Beta'] < advanced_settings['beta_min']:
            continue
        if advanced_settings.get('beta_max') and stock['Beta'] > advanced_settings['beta_max']:
            continue
        
        filtered_stocks.append(stock)
    
    return filtered_stocks

def calculate_technical_signal(stock: Dict[str, Any]) -> float:
    """Calculate technical analysis signal (0-1)"""
    # Simple technical scoring based on RSI and price momentum
    rsi_score = 0.5
    if stock['RSI'] < 30:  # Oversold - potential buy
        rsi_score = 0.8
    elif stock['RSI'] > 70:  # Overbought - potential sell
        rsi_score = 0.2
    elif 40 <= stock['RSI'] <= 60:  # Neutral zone
        rsi_score = 0.6
    
    # Price momentum score
    momentum_score = min(1.0, max(0.0, (stock['Change %'] + 10) / 20))
    
    # Volume factor
    volume_factor = min(1.0, stock['Volume'] / 10000000)  # Normalize volume
    
    return (rsi_score * 0.5 + momentum_score * 0.3 + volume_factor * 0.2)

def calculate_fundamental_signal(stock: Dict[str, Any]) -> float:
    """Calculate fundamental analysis signal (0-1)"""
    # P/E scoring (lower is better for value)
    pe_score = 0.5
    if stock['P/E'] < 15:
        pe_score = 0.8
    elif stock['P/E'] > 40:
        pe_score = 0.2
    
    # ROE scoring (higher is better)
    roe_score = min(1.0, max(0.0, stock['ROE'] / 30))
    
    # Debt/Equity scoring (lower is better)
    debt_score = max(0.0, 1.0 - stock['Debt/Equity'] / 2)
    
    # Growth scoring
    growth_score = min(1.0, max(0.0, (stock['Revenue Growth'] + 20) / 50))
    
    return (pe_score * 0.3 + roe_score * 0.3 + debt_score * 0.2 + growth_score * 0.2)

def calculate_sentiment_signal(stock: Dict[str, Any]) -> float:
    """Calculate sentiment analysis signal (0-1)"""
    # Simple sentiment scoring
    sentiment_score = stock['Sentiment']
    
    # Add some volatility consideration (high volatility can indicate sentiment extremes)
    volatility_factor = min(1.0, stock['Volatility'] / 3)
    
    return sentiment_score * (1 + volatility_factor * 0.2)

def calculate_catalyst_signal(stock: Dict[str, Any], db_manager = None) -> float:
    """Calculate catalyst impact signal (0-1)"""
    # Placeholder catalyst scoring
    # In real implementation, this would query the catalyst database
    import random
    
    # Random catalyst score for demo - in reality this would check for recent catalysts
    # affecting this stock from the database
    base_catalyst_score = random.uniform(0.3, 0.7)
    
    # Factor in volatility (higher volatility might indicate catalyst activity)
    volatility_boost = min(0.3, stock['Volatility'] / 10)
    
    return min(1.0, base_catalyst_score + volatility_boost)