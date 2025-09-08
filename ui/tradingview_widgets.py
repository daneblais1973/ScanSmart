"""
TradingView Widgets for embedding free charts and tickers
No API key or account required - uses public TradingView embed widgets
"""
import streamlit as st
from typing import List, Dict, Optional

class TradingViewWidgets:
    """Free TradingView widgets with no account required"""
    
    @staticmethod
    def ticker_tape(symbols: List[str] = None, theme: str = "light") -> str:
        """Create TradingView ticker tape widget"""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ", "IWM"]
        
        symbols_str = '","'.join(symbols)
        
        widget_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {{
          "symbols": [
            {{
              "proName": "FOREXCOM:SPXUSD",
              "title": "S&P 500"
            }},
            {{
              "proName": "FOREXCOM:NSXUSD", 
              "title": "NASDAQ 100"
            }},
            {{
              "proName": "FX_IDC:EURUSD",
              "title": "EUR/USD"
            }},
            {{
              "proName": "BITSTAMP:BTCUSD",
              "title": "Bitcoin"
            }},
            {{
              "proName": "BITSTAMP:ETHUSD",
              "title": "Ethereum"
            }}
          ],
          "showSymbolLogo": true,
          "colorTheme": "{theme}",
          "isTransparent": false,
          "displayMode": "adaptive",
          "locale": "en"
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        return widget_html
    
    @staticmethod
    def mini_chart(symbol: str = "AAPL", theme: str = "light", width: int = 350, height: int = 220) -> str:
        """Create TradingView mini chart widget"""
        
        widget_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
          {{
          "symbol": "{symbol}",
          "width": {width},
          "height": {height},
          "locale": "en",
          "dateRange": "12M",
          "colorTheme": "{theme}",
          "trendLineColor": "rgba(41, 98, 255, 1)",
          "underLineColor": "rgba(41, 98, 255, 0.3)",
          "underLineBottomColor": "rgba(41, 98, 255, 0)",
          "isTransparent": false,
          "autosize": false,
          "largeChartUrl": ""
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        return widget_html
    
    @staticmethod
    def market_overview(theme: str = "light", width: int = 1000, height: int = 400) -> str:
        """Create TradingView market overview widget"""
        
        widget_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
          {{
          "colorTheme": "{theme}",
          "dateRange": "12M",
          "showChart": true,
          "locale": "en",
          "width": "{width}",
          "height": "{height}",
          "largeChartUrl": "",
          "isTransparent": false,
          "showSymbolLogo": true,
          "showFloatingTooltip": false,
          "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
          "plotLineColorFalling": "rgba(41, 98, 255, 1)",
          "gridLineColor": "rgba(240, 243, 250, 0)",
          "scaleFontColor": "rgba(106, 109, 120, 1)",
          "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
          "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
          "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
          "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
          "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
          "tabs": [
            {{
              "title": "Indices",
              "symbols": [
                {{
                  "s": "FOREXCOM:SPXUSD",
                  "d": "S&P 500"
                }},
                {{
                  "s": "FOREXCOM:NSXUSD",
                  "d": "NASDAQ 100"
                }},
                {{
                  "s": "FOREXCOM:DJI",
                  "d": "Dow 30"
                }},
                {{
                  "s": "INDEX:NKY",
                  "d": "Nikkei 225"
                }},
                {{
                  "s": "INDEX:DAX",
                  "d": "DAX Index"
                }},
                {{
                  "s": "FOREXCOM:UKXGBP",
                  "d": "UK 100"
                }}
              ],
              "originalTitle": "Indices"
            }},
            {{
              "title": "Futures",
              "symbols": [
                {{
                  "s": "CME_MINI:ES1!",
                  "d": "S&P 500"
                }},
                {{
                  "s": "CME:6E1!",
                  "d": "Euro"
                }},
                {{
                  "s": "COMEX:GC1!",
                  "d": "Gold"
                }},
                {{
                  "s": "NYMEX:CL1!",
                  "d": "Crude Oil"
                }},
                {{
                  "s": "NYMEX:NG1!",
                  "d": "Natural Gas"
                }},
                {{
                  "s": "CBOT:ZC1!",
                  "d": "Corn"
                }}
              ],
              "originalTitle": "Futures"
            }},
            {{
              "title": "Bonds",
              "symbols": [
                {{
                  "s": "CME:GE1!",
                  "d": "Eurodollar"
                }},
                {{
                  "s": "CBOT:ZB1!",
                  "d": "T-Bond"
                }},
                {{
                  "s": "CBOT:UB1!",
                  "d": "Ultra T-Bond"
                }},
                {{
                  "s": "EUREX:FGBL1!",
                  "d": "Euro Bund"
                }},
                {{
                  "s": "EUREX:FBTP1!",
                  "d": "Euro BTP"
                }},
                {{
                  "s": "EUREX:FGBM1!",
                  "d": "Euro BOBL"
                }}
              ],
              "originalTitle": "Bonds"
            }},
            {{
              "title": "Forex",
              "symbols": [
                {{
                  "s": "FX:EURUSD",
                  "d": "EUR/USD"
                }},
                {{
                  "s": "FX:GBPUSD",
                  "d": "GBP/USD"
                }},
                {{
                  "s": "FX:USDJPY",
                  "d": "USD/JPY"
                }},
                {{
                  "s": "FX:USDCHF",
                  "d": "USD/CHF"
                }},
                {{
                  "s": "FX:AUDUSD",
                  "d": "AUD/USD"
                }},
                {{
                  "s": "FX:USDCAD",
                  "d": "USD/CAD"
                }}
              ],
              "originalTitle": "Forex"
            }}
          ]
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        return widget_html
    
    @staticmethod
    def economic_calendar(theme: str = "light", width: int = 1000, height: int = 600) -> str:
        """Create TradingView economic calendar widget"""
        
        widget_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
          {{
          "colorTheme": "{theme}",
          "isTransparent": false,
          "width": "{width}",
          "height": "{height}",
          "locale": "en",
          "importanceFilter": "-1,0,1",
          "currencyFilter": "USD,EUR,ITL,NZD,CHF,AUD,RUB,JPY,GBP,CAD,CNY"
          }}
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        return widget_html

def render_tradingview_widgets():
    """Render TradingView widgets in Streamlit"""
    
    st.markdown("### 📈 Live Market Data (TradingView)")
    
    # Ticker tape
    st.markdown("#### Market Overview")
    ticker_tape = TradingViewWidgets.ticker_tape(theme="light")
    st.components.v1.html(ticker_tape, height=100)
    
    # Market overview in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### S&P 500 Chart")
        spx_chart = TradingViewWidgets.mini_chart("SPX", theme="light")
        st.components.v1.html(spx_chart, height=250)
    
    with col2:
        st.markdown("#### NASDAQ Chart")
        nasdaq_chart = TradingViewWidgets.mini_chart("NASDAQ:NDX", theme="light") 
        st.components.v1.html(nasdaq_chart, height=250)
    
    # Full market overview
    st.markdown("#### Global Markets")
    market_overview = TradingViewWidgets.market_overview(theme="light", width=1000, height=400)
    st.components.v1.html(market_overview, height=450)
    
    # Economic calendar
    st.markdown("#### Economic Calendar")
    economic_calendar = TradingViewWidgets.economic_calendar(theme="light", width=1000, height=500)
    st.components.v1.html(economic_calendar, height=550)