import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from ui.custom_ticker_input import ticker_manager


def render_watchlist_section():
    """Render watchlist section with custom tickers and quick stats"""
    
    # Initialize session state if missing
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = set()
    if 'ticker_watchlist' not in st.session_state:
        st.session_state.ticker_watchlist = []
    
    # Get active tickers from ticker manager
    active_tickers = ticker_manager.get_active_tickers()
    
    if not active_tickers:
        # Show empty state with call to action
        st.markdown("""
        ### 📋 Your Watchlist
        <div style='background: rgba(55, 65, 81, 0.6); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'>
            <p style='color: #cbd5e1; margin-bottom: 15px;'>Your watchlist is empty</p>
            <p style='color: #94a3b8; font-size: 14px;'>Add stocks to monitor for catalyst opportunities</p>
            <p style='color: #64748b; font-size: 12px;'>Go to Configuration → Custom Stock Selection to add tickers</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display watchlist header with stats
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📋 Your Watchlist")
    
    with col2:
        st.metric("Tracked Stocks", len(active_tickers))
    
    with col3:
        # Quick action to add more stocks
        if st.button("➕ Add More", key="quick_add_stocks"):
            st.session_state.active_tab = "Config"
            st.rerun()
    
    # Get live data for watchlist tickers
    try:
        watchlist_data = []
        
        # Limit to first 10 tickers for performance
        display_tickers = active_tickers[:10]
        
        for ticker in display_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="2d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    
                    watchlist_data.append({
                        'Ticker': ticker,
                        'Price': f"${current_price:.2f}",
                        'Change': f"{change:+.2f}",
                        'Change %': f"{change_pct:+.1f}%",
                        'Volume': f"{volume/1000000:.1f}M" if volume > 1000000 else f"{volume/1000:.0f}K",
                        'Market Cap': info.get('marketCap', 'N/A'),
                        'Sector': info.get('sector', 'N/A')[:15] + '...' if len(info.get('sector', '')) > 15 else info.get('sector', 'N/A')
                    })
                    
            except Exception as e:
                # Add placeholder for failed tickers
                watchlist_data.append({
                    'Ticker': ticker,
                    'Price': 'N/A',
                    'Change': 'N/A',
                    'Change %': 'N/A',
                    'Volume': 'N/A',
                    'Market Cap': 'N/A',
                    'Sector': 'N/A'
                })
        
        if watchlist_data:
            # Create DataFrame
            df = pd.DataFrame(watchlist_data)
            
            # Style the dataframe
            def style_watchlist(val):
                """Style function for watchlist display"""
                if isinstance(val, str):
                    if val.startswith('+'):
                        return 'color: #10b981'  # Green for positive
                    elif val.startswith('-'):
                        return 'color: #ef4444'  # Red for negative
                return ''
            
            # Display the watchlist table
            st.markdown("""
            <style>
            .watchlist-table {
                background: rgba(30, 41, 59, 0.8);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                df.style.applymap(style_watchlist, subset=['Change', 'Change %']),
                use_container_width=True,
                hide_index=True
            )
            
            # Show truncation notice if needed
            if len(active_tickers) > 10:
                st.info(f"📊 Showing first 10 of {len(active_tickers)} stocks in your watchlist")
        
    except Exception as e:
        st.error(f"Error loading watchlist data: {e}")
        
        # Show basic list without live data
        basic_df = pd.DataFrame({
            'Ticker': active_tickers[:10],
            'Status': ['Monitoring'] * min(len(active_tickers), 10)
        })
        
        st.dataframe(basic_df, use_container_width=True, hide_index=True)
        st.warning("📶 Live market data temporarily unavailable - showing basic watchlist")


def render_watchlist_performance_summary():
    """Render a summary of watchlist performance"""
    active_tickers = ticker_manager.get_active_tickers()
    
    if not active_tickers:
        return
    
    try:
        # Calculate simple performance metrics
        positive_count = 0
        negative_count = 0
        total_volume = 0
        
        for ticker in active_tickers[:10]:  # Limit for performance
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                
                if not hist.empty and len(hist) > 1:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    
                    if current > prev:
                        positive_count += 1
                    else:
                        negative_count += 1
                    
                    if 'Volume' in hist:
                        total_volume += hist['Volume'].iloc[-1]
                        
            except:
                continue
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Gainers", positive_count)
        
        with col2:
            st.metric("📉 Decliners", negative_count)
        
        with col3:
            volume_str = f"{total_volume/1000000:.0f}M" if total_volume > 1000000 else f"{total_volume/1000:.0f}K"
            st.metric("📊 Total Volume", volume_str)
            
    except Exception as e:
        st.warning("Unable to calculate watchlist performance summary")