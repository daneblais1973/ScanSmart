"""
Options calculator and Greeks analysis UI component
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)

def render_options_calculator():
    """Render the options calculator interface"""
    st.title("🎯 Options Calculator & Greeks Analysis")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Options Pricing", 
        "📈 Greeks Analysis", 
        "⚡ Strategy Builder",
        "📋 Options Chain"
    ])
    
    with tab1:
        render_options_pricing()
        
    with tab2:
        render_greeks_analysis()
        
    with tab3:
        render_strategy_builder()
        
    with tab4:
        render_options_chain()

def render_options_pricing():
    """Render options pricing calculator"""
    st.subheader("🔢 Black-Scholes Options Pricing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Input Parameters")
        
        # Stock parameters
        stock_price = st.number_input("Current Stock Price ($):", min_value=0.01, value=100.0, step=0.01)
        strike_price = st.number_input("Strike Price ($):", min_value=0.01, value=100.0, step=0.01)
        
        # Time parameters
        days_to_expiry = st.number_input("Days to Expiration:", min_value=1, value=30, step=1)
        
        # Market parameters
        volatility = st.slider("Volatility (%):", min_value=1, max_value=200, value=25, step=1)
        risk_free_rate = st.slider("Risk-free Rate (%):", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        dividend_yield = st.slider("Dividend Yield (%):", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        option_type = st.selectbox("Option Type:", ["Call", "Put"])
        
    with col2:
        st.write("### Results")
        
        if st.button("💰 Calculate Option Price"):
            try:
                # Calculate option price and Greeks
                calculator = OptionsCalculator()
                
                time_to_expiry = days_to_expiry / 365.0
                vol_decimal = volatility / 100.0
                rate_decimal = risk_free_rate / 100.0
                div_decimal = dividend_yield / 100.0
                
                if option_type == "Call":
                    price = calculator.black_scholes_call(
                        stock_price, strike_price, time_to_expiry, 
                        rate_decimal, vol_decimal, div_decimal
                    )
                    greeks = calculator.calculate_call_greeks(
                        stock_price, strike_price, time_to_expiry,
                        rate_decimal, vol_decimal, div_decimal
                    )
                else:
                    price = calculator.black_scholes_put(
                        stock_price, strike_price, time_to_expiry,
                        rate_decimal, vol_decimal, div_decimal
                    )
                    greeks = calculator.calculate_put_greeks(
                        stock_price, strike_price, time_to_expiry,
                        rate_decimal, vol_decimal, div_decimal
                    )
                
                # Display results
                st.metric("Option Price", f"${price:.2f}")
                
                # Greeks
                st.write("#### Greeks")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.metric("Delta", f"{greeks['delta']:.4f}")
                    st.metric("Gamma", f"{greeks['gamma']:.4f}")
                    st.metric("Vega", f"{greeks['vega']:.4f}")
                
                with col_g2:
                    st.metric("Theta", f"{greeks['theta']:.4f}")
                    st.metric("Rho", f"{greeks['rho']:.4f}")
                
                # Moneyness
                moneyness = stock_price / strike_price
                if moneyness > 1.05:
                    st.success("🟢 In-the-Money")
                elif moneyness < 0.95:
                    st.error("🔴 Out-of-the-Money")
                else:
                    st.warning("🟡 At-the-Money")
                
            except Exception as e:
                st.error(f"Error calculating option price: {e}")
                
    # Sensitivity Analysis
    st.write("### 📊 Sensitivity Analysis")
    
    sensitivity_type = st.selectbox("Analyze sensitivity to:", 
                                   ["Stock Price", "Volatility", "Time to Expiry"])
    
    if st.button("📈 Run Sensitivity Analysis"):
        try:
            calculator = OptionsCalculator()
            
            if sensitivity_type == "Stock Price":
                # Price sensitivity
                price_range = np.linspace(stock_price * 0.7, stock_price * 1.3, 50)
                option_prices = []
                deltas = []
                
                for s in price_range:
                    time_to_expiry = days_to_expiry / 365.0
                    vol_decimal = volatility / 100.0
                    rate_decimal = risk_free_rate / 100.0
                    div_decimal = dividend_yield / 100.0
                    
                    if option_type == "Call":
                        opt_price = calculator.black_scholes_call(
                            s, strike_price, time_to_expiry, 
                            rate_decimal, vol_decimal, div_decimal
                        )
                        greeks = calculator.calculate_call_greeks(
                            s, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                    else:
                        opt_price = calculator.black_scholes_put(
                            s, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                        greeks = calculator.calculate_put_greeks(
                            s, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                    
                    option_prices.append(opt_price)
                    deltas.append(greeks['delta'])
                
                # Plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=price_range, 
                    y=option_prices,
                    mode='lines',
                    name='Option Price',
                    line=dict(color='#00a651', width=3)
                ))
                
                # Add current price line
                fig.add_vline(
                    x=stock_price, 
                    line_dash="dash", 
                    line_color="yellow",
                    annotation_text="Current Price"
                )
                
                fig.update_layout(
                    title=f"{option_type} Option Price vs Stock Price",
                    xaxis_title="Stock Price ($)",
                    yaxis_title="Option Price ($)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif sensitivity_type == "Volatility":
                # Volatility sensitivity
                vol_range = np.linspace(5, 100, 50)
                option_prices = []
                vegas = []
                
                for v in vol_range:
                    time_to_expiry = days_to_expiry / 365.0
                    vol_decimal = v / 100.0
                    rate_decimal = risk_free_rate / 100.0
                    div_decimal = dividend_yield / 100.0
                    
                    if option_type == "Call":
                        opt_price = calculator.black_scholes_call(
                            stock_price, strike_price, time_to_expiry, 
                            rate_decimal, vol_decimal, div_decimal
                        )
                        greeks = calculator.calculate_call_greeks(
                            stock_price, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                    else:
                        opt_price = calculator.black_scholes_put(
                            stock_price, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                        greeks = calculator.calculate_put_greeks(
                            stock_price, strike_price, time_to_expiry,
                            rate_decimal, vol_decimal, div_decimal
                        )
                    
                    option_prices.append(opt_price)
                    vegas.append(greeks['vega'])
                
                # Plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=vol_range, 
                    y=option_prices,
                    mode='lines',
                    name='Option Price',
                    line=dict(color='#00a651', width=3)
                ))
                
                # Add current volatility line
                fig.add_vline(
                    x=volatility, 
                    line_dash="dash", 
                    line_color="yellow",
                    annotation_text="Current Vol"
                )
                
                fig.update_layout(
                    title=f"{option_type} Option Price vs Volatility",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Option Price ($)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error running sensitivity analysis: {e}")

def render_greeks_analysis():
    """Render Greeks analysis interface"""
    st.subheader("📊 Options Greeks Analysis")
    
    # Greeks heatmap
    st.write("### Greeks Heatmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        greek_symbol = st.text_input("Stock Symbol:", value="AAPL")
        greek_strike = st.number_input("Strike Price:", value=150.0)
        
    with col2:
        greek_expiry = st.selectbox("Expiration:", ["1 week", "2 weeks", "1 month", "3 months"])
        greek_type = st.selectbox("Greeks to Display:", ["Delta", "Gamma", "Vega", "Theta"])
    
    if st.button("📈 Generate Greeks Analysis"):
        try:
            # Get current stock price
            ticker = yf.Ticker(greek_symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            calculator = OptionsCalculator()
            
            # Create price and volatility ranges
            price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
            vol_range = np.linspace(0.1, 1.0, 20)
            
            # Calculate Greeks grid
            greeks_grid = []
            
            expiry_days = {"1 week": 7, "2 weeks": 14, "1 month": 30, "3 months": 90}[greek_expiry]
            time_to_expiry = expiry_days / 365.0
            
            for vol in vol_range:
                row = []
                for price in price_range:
                    greeks = calculator.calculate_call_greeks(
                        price, greek_strike, time_to_expiry, 0.05, vol, 0.0
                    )
                    row.append(greeks[greek_type.lower()])
                greeks_grid.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=greeks_grid,
                x=price_range,
                y=vol_range * 100,  # Convert to percentage
                colorscale='RdYlBu',
                hovertemplate=f'{greek_type}: %{{z:.4f}}<br>Price: $%{{x:.2f}}<br>Vol: %{{y:.1f}}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{greek_type} Analysis for {greek_symbol} (Strike: ${greek_strike})",
                xaxis_title="Stock Price ($)",
                yaxis_title="Volatility (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            
            # Add current price and volatility lines
            fig.add_vline(x=current_price, line_dash="dash", line_color="yellow")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating Greeks analysis: {e}")

def render_strategy_builder():
    """Render options strategy builder"""
    st.subheader("⚡ Options Strategy Builder")
    
    # Strategy selection
    strategy_type = st.selectbox("Select Strategy:", [
        "Long Call", "Long Put", "Covered Call", "Protective Put",
        "Bull Call Spread", "Bear Put Spread", "Iron Condor", "Straddle", "Strangle"
    ])
    
    # Strategy parameters
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_symbol = st.text_input("Symbol:", value="AAPL", key="strategy_symbol")
        strategy_quantity = st.number_input("Quantity:", min_value=1, value=1, step=1, key="strategy_quantity")
        
    with col2:
        strategy_price = st.number_input("Current Stock Price:", value=150.0, key="strategy_price")
        
    # Strategy-specific parameters
    if strategy_type in ["Long Call", "Long Put"]:
        strike1 = st.number_input("Strike Price:", value=150.0, key="single_strike")
        expiry_days = st.number_input("Days to Expiry:", value=30, key="single_expiry")
        
    elif strategy_type in ["Bull Call Spread", "Bear Put Spread"]:
        strike1 = st.number_input("Lower Strike:", value=145.0, key="spread_strike1")
        strike2 = st.number_input("Higher Strike:", value=155.0, key="spread_strike2")
        expiry_days = st.number_input("Days to Expiry:", value=30, key="spread_expiry")
        
    elif strategy_type == "Iron Condor":
        put_strike_low = st.number_input("Put Strike (Low):", value=140.0, key="condor_put_low")
        put_strike_high = st.number_input("Put Strike (High):", value=145.0, key="condor_put_high")
        call_strike_low = st.number_input("Call Strike (Low):", value=155.0, key="condor_call_low")
        call_strike_high = st.number_input("Call Strike (High):", value=160.0, key="condor_call_high")
        expiry_days = st.number_input("Days to Expiry:", value=30, key="condor_expiry")
        
    if st.button("📊 Analyze Strategy"):
        try:
            calculator = OptionsCalculator()
            strategy_analyzer = OptionsStrategyAnalyzer(calculator)
            
            if strategy_type == "Long Call":
                result = strategy_analyzer.analyze_long_call(
                    strategy_price, strike1, expiry_days/365.0, 0.05, 0.25
                )
            elif strategy_type == "Long Put":
                result = strategy_analyzer.analyze_long_put(
                    strategy_price, strike1, expiry_days/365.0, 0.05, 0.25
                )
            elif strategy_type == "Bull Call Spread":
                result = strategy_analyzer.analyze_bull_call_spread(
                    strategy_price, strike1, strike2, expiry_days/365.0, 0.05, 0.25
                )
            
            if result:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Net Premium", f"${result.get('net_premium', 0):.2f}")
                    st.metric("Max Profit", f"${result.get('max_profit', 0):.2f}")
                    
                with col2:
                    st.metric("Max Loss", f"${result.get('max_loss', 0):.2f}")
                    st.metric("Break-even", f"${result.get('breakeven', 0):.2f}")
                    
                with col3:
                    st.metric("Profit Probability", f"{result.get('profit_prob', 0):.1%}")
                
                # P&L diagram
                if 'price_range' in result and 'pnl' in result:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=result['price_range'],
                        y=result['pnl'],
                        mode='lines',
                        name='P&L',
                        line=dict(color='#00a651', width=3)
                    ))
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
                    
                    # Add current price line
                    fig.add_vline(x=strategy_price, line_dash="dash", line_color="yellow")
                    
                    fig.update_layout(
                        title=f"{strategy_type} P&L Diagram",
                        xaxis_title="Stock Price at Expiration",
                        yaxis_title="Profit/Loss ($)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error analyzing strategy: {e}")

def render_options_chain():
    """Render live options chain"""
    st.subheader("📋 Live Options Chain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chain_symbol = st.text_input("Symbol for Options Chain:", value="AAPL")
        
    with col2:
        if st.button("🔍 Get Options Chain"):
            st.session_state.fetch_options_chain = True
    
    if getattr(st.session_state, 'fetch_options_chain', False):
        try:
            # Get options chain data
            ticker = yf.Ticker(chain_symbol)
            
            # Get available expiration dates
            expiry_dates = ticker.options
            
            if expiry_dates:
                selected_expiry = st.selectbox("Select Expiration Date:", expiry_dates)
                
                # Get options chain for selected expiry
                options = ticker.option_chain(selected_expiry)
                calls = options.calls
                puts = options.puts
                
                # Display calls and puts side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 📞 Calls")
                    if not calls.empty:
                        # Select important columns
                        calls_display = calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                        calls_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
                        calls_display['IV'] = calls_display['IV'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(calls_display, use_container_width=True, height=400)
                
                with col2:
                    st.write("### 📱 Puts")
                    if not puts.empty:
                        # Select important columns
                        puts_display = puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                        puts_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
                        puts_display['IV'] = puts_display['IV'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(puts_display, use_container_width=True, height=400)
                
                # Implied volatility smile
                if not calls.empty and not puts.empty:
                    st.write("### 📈 Implied Volatility Smile")
                    
                    fig = go.Figure()
                    
                    # Calls IV
                    fig.add_trace(go.Scatter(
                        x=calls['strike'],
                        y=calls['impliedVolatility'] * 100,
                        mode='markers+lines',
                        name='Calls',
                        marker=dict(color='green', size=6)
                    ))
                    
                    # Puts IV
                    fig.add_trace(go.Scatter(
                        x=puts['strike'],
                        y=puts['impliedVolatility'] * 100,
                        mode='markers+lines',
                        name='Puts',
                        marker=dict(color='red', size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"{chain_symbol} Implied Volatility Smile ({selected_expiry})",
                        xaxis_title="Strike Price",
                        yaxis_title="Implied Volatility (%)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning(f"No options data available for {chain_symbol}")
                
        except Exception as e:
            st.error(f"Error fetching options chain: {e}")

class OptionsCalculator:
    """Black-Scholes options calculator"""
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0):
        """Calculate call option price using Black-Scholes formula"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0):
        """Calculate put option price using Black-Scholes formula"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return put_price
    
    def calculate_call_greeks(self, S, K, T, r, sigma, q=0):
        """Calculate Greeks for call option"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        delta = np.exp(-q*T) * norm.cdf(d1)
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q*T) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2) 
                 + q * S * np.exp(-q*T) * norm.cdf(d1)) / 365
        vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    
    def calculate_put_greeks(self, S, K, T, r, sigma, q=0):
        """Calculate Greeks for put option"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        delta = -np.exp(-q*T) * norm.cdf(-d1)
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q*T) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2) 
                 - q * S * np.exp(-q*T) * norm.cdf(-d1)) / 365
        vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

class OptionsStrategyAnalyzer:
    """Options strategy analyzer"""
    
    def __init__(self, calculator: OptionsCalculator):
        self.calculator = calculator
    
    def analyze_long_call(self, S, K, T, r, sigma):
        """Analyze long call strategy"""
        call_price = self.calculator.black_scholes_call(S, K, T, r, sigma)
        
        # P&L analysis
        price_range = np.linspace(S * 0.7, S * 1.3, 100)
        pnl = []
        
        for price in price_range:
            if price > K:
                profit = price - K - call_price
            else:
                profit = -call_price
            pnl.append(profit)
        
        max_profit = float('inf')  # Unlimited upside
        max_loss = -call_price
        breakeven = K + call_price
        profit_prob = norm.cdf((np.log(breakeven/S) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
        
        return {
            'net_premium': -call_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_prob': 1 - profit_prob,
            'price_range': price_range.tolist(),
            'pnl': pnl
        }
    
    def analyze_long_put(self, S, K, T, r, sigma):
        """Analyze long put strategy"""
        put_price = self.calculator.black_scholes_put(S, K, T, r, sigma)
        
        # P&L analysis
        price_range = np.linspace(S * 0.7, S * 1.3, 100)
        pnl = []
        
        for price in price_range:
            if price < K:
                profit = K - price - put_price
            else:
                profit = -put_price
            pnl.append(profit)
        
        max_profit = K - put_price  # Max when stock goes to 0
        max_loss = -put_price
        breakeven = K - put_price
        profit_prob = norm.cdf((np.log(breakeven/S) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
        
        return {
            'net_premium': -put_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_prob': profit_prob,
            'price_range': price_range.tolist(),
            'pnl': pnl
        }
    
    def analyze_bull_call_spread(self, S, K1, K2, T, r, sigma):
        """Analyze bull call spread strategy"""
        long_call = self.calculator.black_scholes_call(S, K1, T, r, sigma)
        short_call = self.calculator.black_scholes_call(S, K2, T, r, sigma)
        net_premium = long_call - short_call
        
        # P&L analysis
        price_range = np.linspace(S * 0.7, S * 1.3, 100)
        pnl = []
        
        for price in price_range:
            long_pnl = max(price - K1, 0) - long_call
            short_pnl = short_call - max(price - K2, 0)
            total_pnl = long_pnl + short_pnl
            pnl.append(total_pnl)
        
        max_profit = K2 - K1 - net_premium
        max_loss = -net_premium
        breakeven = K1 + net_premium
        
        return {
            'net_premium': -net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_prob': 0.5,  # Simplified
            'price_range': price_range.tolist(),
            'pnl': pnl
        }