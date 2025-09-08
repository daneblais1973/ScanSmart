import streamlit as st
import pandas as pd
import logging
from typing import List, Set, Dict, Any
import re

logger = logging.getLogger(__name__)

class CustomTickerManager:
    """Manager for custom ticker input and watchlist functionality"""
    
    def __init__(self):
        # Initialize session state for custom tickers
        if 'custom_tickers' not in st.session_state:
            st.session_state.custom_tickers = set()
        
        if 'ticker_watchlist' not in st.session_state:
            st.session_state.ticker_watchlist = []
        
        # Default popular tickers for quick selection
        self.popular_tickers = {
            'Large Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Consumer': ['KO', 'PG', 'WMT', 'HD', 'MCD', 'NKE'],
            'Small Cap Growth': ['ROKU', 'DKNG', 'PLTR', 'RBLX', 'COIN'],
            'Biotech': ['GILD', 'BIIB', 'REGN', 'VRTX', 'MRNA'],
            'Crypto/Fintech': ['COIN', 'PYPL', 'SQ', 'HOOD', 'SOFI']
        }
        
        # Validation patterns
        self.ticker_pattern = re.compile(r'^[A-Z]{1,5}$')
    
    def render_ticker_input_section(self) -> None:
        """Render the custom ticker input and management section"""
        st.markdown("## 🎯 Custom Stock Selection")
        st.markdown("**Add your own stocks to monitor for catalyst opportunities**")
        
        # Custom ticker input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_ticker = st.text_input(
                "Enter Stock Ticker (e.g., AAPL, TSLA)", 
                placeholder="Type ticker symbol...",
                help="Enter 1-5 letter stock symbols (US markets)"
            ).upper().strip()
        
        with col2:
            if st.button("➕ Add Ticker", type="primary"):
                if self.add_custom_ticker(custom_ticker):
                    st.success(f"✅ {custom_ticker} added!")
                    st.rerun()
        
        # Popular ticker categories
        st.markdown("### 🔥 Popular Categories")
        
        # Create columns for category selection
        cols = st.columns(3)
        for i, (category, tickers) in enumerate(self.popular_tickers.items()):
            with cols[i % 3]:
                if st.button(f"📊 {category}", help=f"Add: {', '.join(tickers)}"):
                    added_count = 0
                    for ticker in tickers:
                        if self.add_custom_ticker(ticker, silent=True):
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"✅ Added {added_count} {category} stocks!")
                        st.rerun()
        
        # Current watchlist display
        self.render_current_watchlist()
        
        # Bulk import section
        self.render_bulk_import_section()
    
    def add_custom_ticker(self, ticker: str, silent: bool = False) -> bool:
        """Add a custom ticker with validation"""
        try:
            if not ticker:
                if not silent:
                    st.error("Please enter a ticker symbol")
                return False
            
            # Validate ticker format
            if not self.ticker_pattern.match(ticker):
                if not silent:
                    st.error("Invalid ticker format. Use 1-5 uppercase letters (e.g., AAPL)")
                return False
            
            # Check for duplicates
            if ticker in st.session_state.custom_tickers:
                if not silent:
                    st.warning(f"{ticker} is already in your watchlist")
                return False
            
            # Add to session state
            st.session_state.custom_tickers.add(ticker)
            
            # Update watchlist with additional info
            st.session_state.ticker_watchlist.append({
                'ticker': ticker,
                'added_date': pd.Timestamp.now(),
                'status': 'active'
            })
            
            logger.info(f"Added custom ticker: {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom ticker {ticker}: {e}")
            if not silent:
                st.error(f"Error adding ticker: {e}")
            return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the watchlist"""
        try:
            if ticker in st.session_state.custom_tickers:
                st.session_state.custom_tickers.remove(ticker)
                
                # Remove from watchlist
                st.session_state.ticker_watchlist = [
                    item for item in st.session_state.ticker_watchlist 
                    if item['ticker'] != ticker
                ]
                
                logger.info(f"Removed custom ticker: {ticker}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing ticker {ticker}: {e}")
            return False
    
    def render_current_watchlist(self) -> None:
        """Render the current watchlist with management options"""
        if st.session_state.custom_tickers:
            st.markdown("### 📋 Your Watchlist")
            
            # Convert to dataframe for display
            watchlist_data = []
            for item in st.session_state.ticker_watchlist:
                watchlist_data.append({
                    'Ticker': item['ticker'],
                    'Added': item['added_date'].strftime('%m/%d/%Y'),
                    'Status': '🟢 Active' if item.get('status') == 'active' else '⚪ Inactive'
                })
            
            df = pd.DataFrame(watchlist_data)
            
            # Display with action buttons
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Actions:**")
                
                # Remove ticker selectbox
                ticker_to_remove = st.selectbox(
                    "Remove ticker:",
                    options=[''] + list(st.session_state.custom_tickers),
                    format_func=lambda x: "Select ticker..." if x == '' else x
                )
                
                if ticker_to_remove and st.button("🗑️ Remove", key="remove_ticker"):
                    if self.remove_ticker(ticker_to_remove):
                        st.success(f"✅ Removed {ticker_to_remove}")
                        st.rerun()
                
                # Clear all button
                if len(st.session_state.custom_tickers) > 0:
                    if st.button("🗑️ Clear All", type="secondary"):
                        st.session_state.custom_tickers.clear()
                        st.session_state.ticker_watchlist.clear()
                        st.success("✅ Watchlist cleared!")
                        st.rerun()
        else:
            st.info("📝 Your watchlist is empty. Add some tickers to get started!")
    
    def render_bulk_import_section(self) -> None:
        """Render bulk import functionality"""
        with st.expander("📥 Bulk Import"):
            st.markdown("**Import multiple tickers at once**")
            
            bulk_text = st.text_area(
                "Enter tickers (comma or space separated):",
                placeholder="AAPL, MSFT, GOOGL AMZN TSLA",
                help="Separate tickers with commas or spaces"
            )
            
            if st.button("📥 Import All", key="bulk_import"):
                if bulk_text.strip():
                    # Parse the input
                    import re
                    tickers = re.findall(r'[A-Z]{1,5}', bulk_text.upper())
                    
                    added_count = 0
                    invalid_count = 0
                    duplicate_count = 0
                    
                    for ticker in tickers:
                        if ticker in st.session_state.custom_tickers:
                            duplicate_count += 1
                        elif self.add_custom_ticker(ticker, silent=True):
                            added_count += 1
                        else:
                            invalid_count += 1
                    
                    # Show results
                    if added_count > 0:
                        st.success(f"✅ Added {added_count} tickers")
                    if duplicate_count > 0:
                        st.info(f"ℹ️ Skipped {duplicate_count} duplicates")
                    if invalid_count > 0:
                        st.warning(f"⚠️ {invalid_count} invalid tickers ignored")
                    
                    if added_count > 0:
                        st.rerun()
                else:
                    st.error("Please enter some tickers to import")
    
    def get_active_tickers(self) -> List[str]:
        """Get list of all active tickers for monitoring"""
        return list(st.session_state.custom_tickers)
    
    def get_watchlist_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the watchlist"""
        return {
            'total_tickers': len(st.session_state.custom_tickers),
            'active_tickers': len([t for t in st.session_state.ticker_watchlist if t.get('status') == 'active']),
            'tickers': list(st.session_state.custom_tickers),
            'last_updated': max([t['added_date'] for t in st.session_state.ticker_watchlist]) if st.session_state.ticker_watchlist else None
        }
    
    def render_watchlist_stats(self) -> None:
        """Render watchlist statistics"""
        summary = self.get_watchlist_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tickers", summary['total_tickers'])
        
        with col2:
            st.metric("Active", summary['active_tickers'])
        
        with col3:
            if summary['last_updated']:
                days_since = (pd.Timestamp.now() - summary['last_updated']).days
                st.metric("Days Since Update", days_since)
            else:
                st.metric("Last Updated", "Never")

# Global instance for use in other modules
ticker_manager = CustomTickerManager()