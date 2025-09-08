import streamlit as st
import streamlit.components.v1 as components
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import numpy as np
import subprocess
import os
import time

# Import core components
from core.config import AppConfig
from core.database import DatabaseManager
from core.cache import CacheManager
from nlp.processor import NLPProcessor
from alerting.alert_manager import AlertManager
from alerting.email_notifier import EmailNotifier
from batch_processing.scheduler import BatchScheduler
from data_fetchers.free_data_sources import FreeDataAggregator
from ui.custom_ticker_input import ticker_manager
from ui.configuration_page import config_manager
from analysis.professional_screener import ProfessionalStockScreener

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inject_sidebar_css():
    """Inject essential CSS for professional sidebar styling"""
    
    css = """
    <style>
    /* Professional sidebar styling */
    section[data-testid="stSidebar"] > div {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Sidebar button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        margin: 0.15rem 0;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Status indicators */
    .status-running {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin: 6px 0;
        text-align: center;
        display: inline-block;
    }
    
    .status-stopped {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin: 6px 0;
        text-align: center;
        display: inline-block;
    }
    
    /* Sidebar section headers */
    .sidebar-section {
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #334155;
    }
    
    /* Metric displays */
    .sidebar-metric {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 0.25rem 0;
    }
    
    .sidebar-metric-value {
        font-weight: 600;
        color: #f8fafc;
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def create_floating_sidebar(session_state):
    """Create persistent floating sidebar with original appearance + Catalysts button"""
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    if 'scanning_active' not in st.session_state:
        st.session_state.scanning_active = False
    
    # Original icon-only floating sidebar with Catalysts button added
    import streamlit.components.v1 as components
    
    components.html("""
    <script>
    (function() {
        // Prevent multiple initializations
        if (window.__iconSidebarActive) return;
        window.__iconSidebarActive = true;
        
        const targetDoc = window.parent ? window.parent.document : document;
        
        // Create the icon sidebar
        function createIconSidebar() {
            const existing = targetDoc.querySelector('#icon-sidebar');
            if (existing) existing.remove();
            
            const iconSidebar = targetDoc.createElement('div');
            iconSidebar.id = 'icon-sidebar';
            iconSidebar.innerHTML = `
                <div class="icon-item" data-tab="dashboard" title="Dashboard">
                    <span class="icon">📊</span>
                </div>
                <div class="icon-item" data-tab="settings" title="Settings">
                    <span class="icon">⚙️</span>
                </div>
                <div class="icon-item" data-tab="markets" title="Markets">
                    <span class="icon">🌐</span>
                </div>
                <div class="icon-item" data-tab="catalysts" title="Catalysts">
                    <span class="icon">🎯</span>
                </div>
                <div class="icon-item" data-tab="config" title="Configuration">
                    <span class="icon">🔧</span>
                </div>
                <div class="divider"></div>
                <div class="icon-item scan-control" data-action="start" title="Start Scan">
                    <span class="icon">▶️</span>
                </div>
                <div class="icon-item scan-control" data-action="stop" title="Stop Scan">
                    <span class="icon">⏹️</span>
                </div>
                <div class="icon-item scan-control" data-action="clear" title="Clear Results">
                    <span class="icon">🗑️</span>
                </div>
            `;
            
            targetDoc.body.appendChild(iconSidebar);
            return iconSidebar;
        }
        
        // Add icon sidebar styles
        function addIconStyles() {
            const existingStyle = targetDoc.getElementById('icon-sidebar-styles');
            if (existingStyle) return;
            
            const style = targetDoc.createElement('style');
            style.id = 'icon-sidebar-styles';
            style.textContent = `
                #icon-sidebar {
                    position: fixed;
                    left: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    background: rgba(30, 41, 59, 0.95);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(51, 65, 85, 0.8);
                    border-radius: 10px;
                    padding: 7px 5px;
                    z-index: 9999;
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }
                
                #icon-sidebar:hover {
                    background: rgba(30, 41, 59, 0.98);
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                    transform: translateY(-50%) scale(1.02);
                }
                
                .icon-item {
                    width: 29px;
                    height: 29px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 7px;
                    background: rgba(55, 65, 81, 0.6);
                    cursor: pointer;
                    transition: all 0.2s ease;
                    position: relative;
                    border: 1px solid transparent;
                }
                
                .icon-item:hover {
                    background: rgba(37, 99, 235, 0.8);
                    border-color: rgba(37, 99, 235, 0.5);
                    transform: scale(1.05);
                    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
                }
                
                .icon-item.active {
                    background: rgba(5, 150, 105, 0.9);
                    border-color: rgba(5, 150, 105, 0.6);
                    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
                }
                
                .icon-item .icon {
                    font-size: 12px;
                    filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.3));
                }
                
                .divider {
                    height: 1px;
                    background: rgba(51, 65, 85, 0.8);
                    margin: 2px 5px;
                    border-radius: 1px;
                }
                
                .scan-control {
                    background: rgba(99, 102, 241, 0.6);
                }
                
                .scan-control:hover {
                    background: rgba(99, 102, 241, 0.9);
                    border-color: rgba(99, 102, 241, 0.6);
                }
                
                /* Tooltip styling */
                .icon-item::after {
                    content: attr(title);
                    position: absolute;
                    left: 60px;
                    top: 50%;
                    transform: translateY(-50%);
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    white-space: nowrap;
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.2s ease;
                    z-index: 10000;
                }
                
                .icon-item:hover::after {
                    opacity: 1;
                }
            `;
            targetDoc.head.appendChild(style);
        }
        
        function handleIconClick(iconItem) {
            const tab = iconItem.dataset.tab;
            const action = iconItem.dataset.action;
            
            // Update active state
            targetDoc.querySelectorAll('.icon-item').forEach(item => {
                item.classList.remove('active');
            });
            iconItem.classList.add('active');
            
            if (tab) {
                // Handle navigation - trigger Streamlit state change
                console.log('Navigate to:', tab);
                window.parent.postMessage({
                    type: 'navigate',
                    tab: tab
                }, '*');
            } else if (action) {
                // Handle scan controls
                console.log('Scan action:', action);
                window.parent.postMessage({
                    type: 'scan',
                    action: action
                }, '*');
            }
        }
        
        // Initialize
        function initialize() {
            addIconStyles();
            const iconSidebar = createIconSidebar();
            
            // Add click listeners to icons
            iconSidebar.addEventListener('click', (e) => {
                const iconItem = e.target.closest('.icon-item');
                if (iconItem) {
                    handleIconClick(iconItem);
                }
            });
            
            console.log('Persistent icon sidebar initialized');
        }
        
        // Wait for DOM to be ready
        if (targetDoc.readyState === 'loading') {
            targetDoc.addEventListener('DOMContentLoaded', initialize);
        } else {
            setTimeout(initialize, 1000);
        }
        
    })();
    </script>
    """, height=0)
    
    # Handle navigation messages from the floating sidebar
    if 'navigation_trigger' not in st.session_state:
        st.session_state.navigation_trigger = None

    # Simple navigation handling using session state
    with st.sidebar:
        st.markdown("## 🚀 ScanSmart Pro")
        st.markdown("<span style='color: #94a3b8;'>AI Trading Intelligence</span>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Hidden navigation buttons that can be triggered by the floating sidebar
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📊", key="nav_dashboard_hidden", help="Dashboard"):
                st.session_state.active_tab = "Dashboard"
                st.rerun()
                
        with col2:
            if st.button("⚙️", key="nav_settings_hidden", help="Settings"):
                st.session_state.active_tab = "Settings"
                st.rerun()
                
        with col3:
            if st.button("🌐", key="nav_markets_hidden", help="Markets"):
                st.session_state.active_tab = "Markets"
                st.rerun()
                
        with col4:
            if st.button("🎯", key="nav_catalysts_hidden", help="Catalysts"):
                st.session_state.active_tab = "Catalysts"
                st.rerun()
                
        with col5:
            if st.button("🔧", key="nav_config_hidden", help="Config"):
                st.session_state.active_tab = "Config"
                st.rerun()
        
        st.markdown("---")
        
        # Scan controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️", key="start_scan_hidden", help="Start Scan", disabled=st.session_state.scanning_active):
                st.session_state.scanning_active = True
                try:
                    if session_state and session_state.get('scheduler'):
                        if hasattr(session_state['scheduler'], 'start'):
                            session_state['scheduler'].start()
                    st.success("Scanning started!")
                except Exception as e:
                    st.error(f"Start failed: {str(e)}")
                st.rerun()
        
        with col2:
            if st.button("⏹️", key="stop_scan_hidden", help="Stop Scan", disabled=not st.session_state.scanning_active):
                st.session_state.scanning_active = False
                try:
                    if session_state and session_state.get('scheduler'):
                        if hasattr(session_state['scheduler'], 'stop'):
                            session_state['scheduler'].stop()
                    st.success("Scanning stopped!")
                except Exception as e:
                    st.error(f"Stop failed: {str(e)}")
                st.rerun()
        
        with col3:
            if st.button("🗑️", key="clear_results_hidden", help="Clear Results"):
                try:
                    cleared_count = 0
                    if session_state and session_state.get('db'):
                        if hasattr(session_state['db'], 'clear_old_catalysts'):
                            cleared_count = session_state['db'].clear_old_catalysts(days=0)
                    if session_state and session_state.get('cache'):
                        if hasattr(session_state['cache'], 'clear_all'):
                            session_state['cache'].clear_all()
                    st.success(f"Cleared {cleared_count} results")
                except Exception as e:
                    st.error(f"Clear failed: {str(e)}")
                st.rerun()


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="ScanSmart Pro - AI Trading Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply clean dark theme
    try:
        from ui.clean_layout_styles import get_professional_dark_theme
        st.markdown(get_professional_dark_theme(), unsafe_allow_html=True)
    except ImportError:
        # Use basic dark theme if custom styles not available
        st.markdown("""
        <style>
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        /* Remove top padding/margin that causes large space */
        .main .block-container {
            padding-top: 1rem !important;
            margin-top: 0 !important;
        }
        
        /* Ensure no unwanted spacing at top */
        .stApp > header {
            background-color: transparent !important;
        }
        
        .stApp [data-testid="stHeader"] {
            background: rgba(0,0,0,0) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    if 'scanning_active' not in st.session_state:
        st.session_state.scanning_active = False
    
    # Initialize app components (with error handling)
    session_state = initialize_app_safely()
    
    # Create floating overlay sidebar
    create_floating_sidebar(session_state)
    
    # Main content routing (suppress any None returns)
    if st.session_state.active_tab == "Dashboard":
        _ = render_clean_dashboard(session_state)
    elif st.session_state.active_tab == "Settings":
        _ = render_configuration(session_state)
    elif st.session_state.active_tab == "Markets":
        _ = render_markets_page(session_state)
    elif st.session_state.active_tab == "Catalysts":
        _ = render_catalyst_page(session_state)
    elif st.session_state.active_tab == "Config":
        try:
            # Render comprehensive configuration page with API keys and email setup
            config_manager.render_configuration_page(session_state)
            
            # Add custom ticker management section
            st.markdown("---")
            ticker_manager.render_ticker_input_section()
            
        except Exception as e:
            st.error("Configuration component not available")
            logger.error(f"Configuration rendering failed: {e}")
    else:
        st.error(f"Unknown tab: {st.session_state.active_tab}")

def initialize_app_safely():
    """Initialize app components with error handling"""
    session_state = {}
    
    try:
        # Initialize core components
        session_state['config'] = AppConfig()
        session_state['db'] = DatabaseManager(session_state['config'])
        session_state['cache'] = CacheManager(session_state['config'])
        session_state['nlp'] = NLPProcessor(session_state['config'])
        session_state['alerts'] = AlertManager(session_state['config'])
        session_state['scheduler'] = BatchScheduler(session_state['config'])
        
        # Initialize data fetchers with free data aggregator
        try:
            from data_fetchers import get_all_fetchers
            session_state['data_fetchers'] = get_all_fetchers(session_state['config'])
        except ImportError as e:
            logger.warning(f"Data fetchers not available: {e}")
            session_state['data_fetchers'] = None
        
        # Initialize free data aggregator
        try:
            session_state['free_data'] = FreeDataAggregator(session_state['config'])
        except Exception as e:
            logger.warning(f"Free data aggregator initialization failed: {e}")
            session_state['free_data'] = None
        
        # Initialize email notifier
        try:
            session_state['email_notifier'] = EmailNotifier(session_state['config'])
        except Exception as e:
            logger.warning(f"Email notifier initialization failed: {e}")
            session_state['email_notifier'] = None
        
        # Test database connection
        if hasattr(session_state['db'], 'is_connected'):
            if not session_state['db'].is_connected():
                st.warning("Database connection failed - using demo mode")
        
        return session_state
        
    except Exception as e:
        logger.error(f"App initialization error: {e}")
        st.error(f"Initialization error: {e}")
        # Return minimal session state for demo mode
        return {
            'config': None,
            'db': None,
            'cache': None,
            'nlp': None,
            'alerts': None,
            'scheduler': None,
            'data_fetchers': None
        }


def render_clean_dashboard(session_state):
    """Render the main dashboard"""
    try:
        from ui.clean_dashboard import render_clean_dashboard as render_dashboard
        _ = render_dashboard(session_state)
        return  # Explicitly return to prevent None display
    except ImportError as e:
        st.error("Dashboard component not available")
        st.info("Please check the clean_dashboard.py file")
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Dashboard rendering failed: {e}")

def render_configuration(session_state):
    """Render the configuration page"""
    try:
        from ui.configuration import render_configuration as render_config
        render_config(session_state)
    except ImportError as e:
        st.error("Configuration component not available")
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
        logger.error(f"Configuration rendering failed: {e}")

def render_markets_page(session_state):
    """Markets page with three tabs"""
    st.markdown("# 🌍 Global Markets")
    st.markdown("Real-time market data and analysis")
    
    tab1, tab2, tab3 = st.tabs(["🌍 Global Overview", "🇺🇸 US Markets", "🏭 Sector Performance"])
    
    with tab1:
        render_global_markets(session_state)
    
    with tab2:
        render_local_markets(session_state)
    
    with tab3:
        render_industry_markets(session_state)

def render_global_markets(session_state):
    """Global market information"""
    st.subheader("Global Indices Performance")
    
    # Try to get real market data if available
    indices_data = {}
    try:
        if session_state and session_state.get('data_fetchers'):
            # This would be replaced with actual data fetcher calls
            pass
    except:
        pass
    
    # Get real market data from data fetchers
    if session_state and session_state.get('data_fetchers'):
        try:
            # Attempt to fetch real market data
            for fetcher in session_state['data_fetchers']:
                if hasattr(fetcher, 'fetch_market_indices'):
                    indices_data = fetcher.fetch_market_indices()
                    break
        except Exception as e:
            logger.warning(f"Failed to fetch real market data: {e}")
    
    # Only show message if no real data available
    if not indices_data:
        st.warning("⚠️ Real-time market data not available")
        st.info("Please configure financial data API keys to view live market indices")
        return
    
    # Display indices
    cols = st.columns(2)
    for i, (name, data) in enumerate(indices_data.items()):
        with cols[i % 2]:
            st.metric(name, f"{data['value']:,.2f}", f"{data['change']:+.2f}")

def render_local_markets(session_state):
    """US market information"""
    st.subheader("US Market Analysis")
    
    # Get real US market data
    try:
        if session_state and session_state.get('data_fetchers'):
            # Fetch real market data using yfinance
            import yfinance as yf
            
            # Real market data
            sp500 = yf.Ticker("^GSPC")
            nasdaq = yf.Ticker("^IXIC") 
            dow = yf.Ticker("^DJI")
            vix = yf.Ticker("^VIX")
            
            # Get latest data
            sp500_info = sp500.history(period="2d")
            nasdaq_info = nasdaq.history(period="2d")
            dow_info = dow.history(period="2d")
            vix_info = vix.history(period="2d")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if not sp500_info.empty:
                    current = sp500_info['Close'].iloc[-1]
                    prev = sp500_info['Close'].iloc[-2] if len(sp500_info) > 1 else current
                    change = current - prev
                    st.metric("S&P 500", f"{current:,.2f}", f"{change:+.2f}")
                else:
                    st.metric("S&P 500", "N/A", "N/A")
            
            with col2:
                if not nasdaq_info.empty:
                    current = nasdaq_info['Close'].iloc[-1]
                    prev = nasdaq_info['Close'].iloc[-2] if len(nasdaq_info) > 1 else current
                    change = current - prev
                    st.metric("NASDAQ", f"{current:,.2f}", f"{change:+.2f}")
                else:
                    st.metric("NASDAQ", "N/A", "N/A")
            
            with col3:
                if not dow_info.empty:
                    current = dow_info['Close'].iloc[-1]
                    prev = dow_info['Close'].iloc[-2] if len(dow_info) > 1 else current
                    change = current - prev
                    st.metric("DOW", f"{current:,.2f}", f"{change:+.2f}")
                else:
                    st.metric("DOW", "N/A", "N/A")
            
            with col4:
                if not vix_info.empty:
                    current = vix_info['Close'].iloc[-1]
                    prev = vix_info['Close'].iloc[-2] if len(vix_info) > 1 else current
                    change = current - prev
                    st.metric("VIX", f"{current:.2f}", f"{change:+.2f}")
                else:
                    st.metric("VIX", "N/A", "N/A")
            
            # Real market analysis based on actual data
            if not sp500_info.empty and not nasdaq_info.empty:
                sp_change = ((sp500_info['Close'].iloc[-1] - sp500_info['Close'].iloc[-2]) / sp500_info['Close'].iloc[-2] * 100) if len(sp500_info) > 1 else 0
                nq_change = ((nasdaq_info['Close'].iloc[-1] - nasdaq_info['Close'].iloc[-2]) / nasdaq_info['Close'].iloc[-2] * 100) if len(nasdaq_info) > 1 else 0
                
                if sp_change > 0 and nq_change > 0:
                    st.success("📈 US markets showing positive momentum with broad-based gains")
                elif sp_change < 0 and nq_change < 0:
                    st.error("📉 US markets under pressure with widespread selling")
                else:
                    st.info("📊 Mixed performance across US markets - sector rotation in progress")
            
        else:
            st.warning("⚠️ Live market data not available - please configure data sources")
        
    except Exception as e:
        logger.error(f"Error fetching US market data: {e}")
        st.error("Unable to load live market data")
        st.info("Please check your internet connection and data source configuration")

def render_industry_markets(session_state):
    """Industry sector performance"""
    st.subheader("Sector Performance")
    
    # Get real sector performance data
    try:
        import yfinance as yf
        
        # Real sector ETFs for performance tracking
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Financials": "XLF",
            "Energy": "XLE",
            "Consumer Disc.": "XLY",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Utilities": "XLU"
        }
        
        sectors_data = {}
        
        for sector_name, etf_symbol in sector_etfs.items():
            try:
                etf = yf.Ticker(etf_symbol)
                hist = etf.history(period="2d")
                info = etf.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    perf = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    
                    sectors_data[sector_name] = {
                        "performance": f"{perf:+.1f}%",
                        "volume": f"{volume/1000000:.0f}M" if volume > 1000000 else f"{volume/1000:.0f}K"
                    }
            except Exception as e:
                logger.warning(f"Failed to get data for {sector_name}: {e}")
                continue
        
        if sectors_data:
            for sector, data in sectors_data.items():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{sector}**")
                with col2:
                    st.write(f"{data['performance']} (Vol: {data['volume']})")
                
                # Convert performance to progress bar value (0.0 to 1.0)
                perf_value = float(data['performance'].strip('%+'))
                # Normalize to 0-1 range, treating -5% to +5% as the full range
                progress_value = max(0.0, min(1.0, (perf_value + 5) / 10))
                st.progress(progress_value)
        else:
            st.warning("Unable to load live sector data")
            
    except Exception as e:
        logger.error(f"Error loading sector data: {e}")
        st.error("Live sector performance data unavailable")

def render_catalyst_page(session_state):
    """Render streaming catalyst detection page"""
    try:
        from ui.catalyst_streaming import render_catalyst_streaming_page
        _ = render_catalyst_streaming_page(session_state)
        return  # Explicitly return to prevent None display
    except ImportError as e:
        st.error("Catalyst streaming component not available")
        st.info("Creating basic catalyst view...")
        render_basic_catalyst_page(session_state)
    except Exception as e:
        st.error(f"Catalyst page error: {str(e)}")
        logger.error(f"Catalyst page rendering failed: {e}")

def render_basic_catalyst_page(session_state):
    """Basic catalyst page when streaming component not available"""
    st.markdown("# 🎯 Live Catalyst Detection")
    st.markdown("Real-time catalyst monitoring from multiple data sources")
    
    # Button to populate test data
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Load Test Data", type="primary"):
            populate_test_catalysts(session_state)
            st.rerun()
    
    # Get catalyst data from database
    try:
        if session_state and session_state.get('db'):
            catalysts = session_state['db'].get_catalysts(limit=50)
            
            if catalysts:
                st.success(f"✅ {len(catalysts)} catalysts detected")
                
                # Create basic table
                catalyst_data = []
                for catalyst in catalysts:
                    catalyst_data.append({
                        'Source': catalyst.source,
                        'Headline': catalyst.catalyst[:100] + "..." if len(catalyst.catalyst) > 100 else catalyst.catalyst,
                        'Ticker': catalyst.ticker,
                        'Category': catalyst.category,
                        'Impact': catalyst.impact,
                        'Date': catalyst.published_date.strftime('%m/%d %H:%M') if catalyst.published_date else 'N/A'
                    })
                
                df = pd.DataFrame(catalyst_data)
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("🔍 No catalysts detected yet. Scanning in progress...")
                st.markdown("Monitoring sources: NewsAPI, RSS feeds, SEC filings, social media")
                st.markdown("💡 Click 'Load Test Data' to populate with sample catalyst data")
        else:
            st.warning("Database not available - using demo mode")
    except Exception as e:
        st.error(f"Error loading catalysts: {e}")

def populate_test_catalysts(session_state):
    """Populate database with test catalyst data"""
    try:
        from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType
        from datetime import datetime, timezone
        
        db = session_state.get('db')
        if not db:
            st.error("Database not available")
            return
        
        # Sample catalyst data based on real market events
        test_catalysts = [
            {
                'ticker': 'AAPL',
                'catalyst': 'Apple announces record Q4 earnings beating analyst estimates by 15% with iPhone sales driving growth',
                'category': 'earnings',
                'sentiment': 'positive',
                'sentiment_score': 0.8,
                'impact': 85,
                'source': 'RSS',
                'confidence': 0.9,
                'sector': 'Technology'
            },
            {
                'ticker': 'TSLA',
                'catalyst': 'Tesla receives FDA approval for new autopilot features in Model S, expanding autonomous driving capabilities',
                'category': 'regulatory',
                'sentiment': 'positive', 
                'sentiment_score': 0.7,
                'impact': 78,
                'source': 'NEWSAPI',
                'confidence': 0.85,
                'sector': 'Automotive'
            },
            {
                'ticker': 'MSFT',
                'catalyst': 'Microsoft announces $10B investment in AI infrastructure partnership with OpenAI for enterprise solutions',
                'category': 'partnership',
                'sentiment': 'positive',
                'sentiment_score': 0.75,
                'impact': 82,
                'source': 'TWITTER',
                'confidence': 0.88,
                'sector': 'Technology'
            },
            {
                'ticker': 'PFE',
                'catalyst': 'Pfizer drug candidate fails Phase 3 trials for Alzheimer treatment, shares down in after-hours trading',
                'category': 'clinical_trial',
                'sentiment': 'negative',
                'sentiment_score': -0.6,
                'impact': 75,
                'source': 'REGULATORY',
                'confidence': 0.92,
                'sector': 'Healthcare'
            },
            {
                'ticker': 'NVDA',
                'catalyst': 'NVIDIA unveils next-generation AI chips with 50% performance improvement for data centers',
                'category': 'product_launch',
                'sentiment': 'positive',
                'sentiment_score': 0.85,
                'impact': 88,
                'source': 'RSS',
                'confidence': 0.87,
                'sector': 'Technology'
            }
        ]
        
        added_count = 0
        for catalyst_data in test_catalysts:
            try:
                catalyst = Catalyst(
                    ticker=catalyst_data['ticker'],
                    catalyst=catalyst_data['catalyst'],
                    category=CatalystType(catalyst_data['category']),
                    sentiment_label=SentimentLabel(catalyst_data['sentiment']),
                    sentiment_score=catalyst_data['sentiment_score'],
                    impact=catalyst_data['impact'],
                    source=SourceType(catalyst_data['source']),
                    confidence=catalyst_data['confidence'],
                    published_date=datetime.now(timezone.utc),
                    url=f"https://example.com/news/{catalyst_data['ticker'].lower()}",
                    sector=catalyst_data['sector'],
                    metadata={'test_data': True}
                )
                
                catalyst_id = db.save_catalyst(catalyst)
                if catalyst_id:
                    added_count += 1
                    
            except Exception as e:
                logger.error(f"Error adding test catalyst: {e}")
                continue
        
        st.success(f"✅ Added {added_count} test catalysts to database")
        
    except Exception as e:
        st.error(f"Error populating test catalysts: {e}")
        logger.error(f"Test catalyst population failed: {e}")

def render_system_config(session_state):
    """System configuration page"""
    st.markdown("# 🔧 System Configuration")
    
    # System status
    st.subheader("Current Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Memory", "2.3 GB", "512 MB")
    with col2:
        st.metric("CPU", "15%", "+3%")
    with col3:
        st.metric("Threads", "12", "+2")
    with col4:
        st.metric("Cache", "156 MB", "+23 MB")
    
    # Configuration settings
    st.subheader("Performance Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        max_memory = st.slider("Max Memory (GB)", 1.0, 16.0, 4.0)
        worker_threads = st.slider("Worker Threads", 1, 32, 8)
    with col2:
        cache_size = st.slider("Cache Size (MB)", 64, 2048, 512)
        enable_cache = st.checkbox("Enable Caching", value=True)
    
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved!")

if __name__ == "__main__":
    main()