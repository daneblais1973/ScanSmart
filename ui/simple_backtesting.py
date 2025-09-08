"""
Simple backtesting UI component
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from analysis.simple_backtester import SimpleBacktester, sma_crossover_strategy, rsi_strategy, bollinger_bands_strategy

def render_simple_backtesting():
    """Render the simple backtesting interface"""
    st.title("📊 Strategy Backtesting")
    st.write("Test your trading strategies with historical data")
    
    # Strategy selection and parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Strategy Configuration")
        
        strategy_name = st.selectbox(
            "Select Strategy:",
            ["SMA Crossover", "RSI Mean Reversion", "Bollinger Bands"]
        )
        
        symbol = st.text_input("Stock Symbol:", value="AAPL")
        
        initial_capital = st.number_input(
            "Initial Capital ($):",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )
        
        # Date range
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=365)
            )
        with col_date2:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now()
            )
        
        # Strategy-specific parameters
        if strategy_name == "SMA Crossover":
            fast_period = st.slider("Fast SMA Period:", 5, 50, 10)
            slow_period = st.slider("Slow SMA Period:", 10, 100, 20)
            strategy_params = {
                'fast_period': fast_period,
                'slow_period': slow_period
            }
            strategy_func = sma_crossover_strategy
            
        elif strategy_name == "RSI Mean Reversion":
            rsi_period = st.slider("RSI Period:", 5, 30, 14)
            oversold = st.slider("Oversold Level:", 10, 40, 30)
            overbought = st.slider("Overbought Level:", 60, 90, 70)
            strategy_params = {
                'rsi_period': rsi_period,
                'oversold': oversold,
                'overbought': overbought
            }
            strategy_func = rsi_strategy
            
        elif strategy_name == "Bollinger Bands":
            bb_period = st.slider("BB Period:", 10, 50, 20)
            num_std = st.slider("Number of Std Dev:", 1.0, 3.0, 2.0, 0.1)
            strategy_params = {
                'period': bb_period,
                'num_std': num_std
            }
            strategy_func = bollinger_bands_strategy
    
    with col2:
        st.write("### Backtest Results")
        
        if st.button("🚀 Run Backtest", type="primary"):
            try:
                with st.spinner("Running backtest..."):
                    backtester = SimpleBacktester(initial_capital=initial_capital)
                    
                    result = backtester.run_backtest(
                        strategy_func=strategy_func,
                        symbol=symbol.upper(),
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        **strategy_params
                    )
                
                if result:
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        color = "green" if result.total_return > 0 else "red"
                        st.metric(
                            "Total Return",
                            f"{result.total_return:.2f}%",
                            delta=f"{result.total_return:.2f}%"
                        )
                        
                    with col2:
                        st.metric(
                            "Annual Return",
                            f"{result.annual_return:.2f}%"
                        )
                        
                    with col3:
                        st.metric(
                            "Max Drawdown",
                            f"{result.max_drawdown:.2f}%"
                        )
                        
                    with col4:
                        st.metric(
                            "Sharpe Ratio",
                            f"{result.sharpe_ratio:.2f}"
                        )
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Trades", f"{result.num_trades}")
                        
                    with col2:
                        st.metric("Win Rate", f"{result.win_rate:.1f}%")
                        
                    with col3:
                        if result.profit_factor == float('inf'):
                            st.metric("Profit Factor", "∞")
                        else:
                            st.metric("Profit Factor", f"{result.profit_factor:.2f}")
                    
                    # Equity curve chart
                    st.write("### 📈 Equity Curve")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve.values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#00a651', width=2)
                    ))
                    
                    # Add buy & hold comparison
                    try:
                        import yfinance as yf
                        
                        benchmark_data = yf.download(
                            symbol.upper(),
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d')
                        )
                        
                        if not benchmark_data.empty:
                            initial_price = benchmark_data['Close'].iloc[0]
                            benchmark_values = (benchmark_data['Close'] / initial_price) * initial_capital
                            
                            fig.add_trace(go.Scatter(
                                x=benchmark_values.index,
                                y=benchmark_values.values,
                                mode='lines',
                                name='Buy & Hold',
                                line=dict(color='#ff6b6b', width=2, dash='dash')
                            ))
                            
                    except Exception as e:
                        st.warning("Could not load benchmark data for comparison")
                    
                    fig.update_layout(
                        title=f"{strategy_name} Strategy Performance",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade history
                    if result.trades:
                        st.write("### 📋 Trade History")
                        
                        trades_df = pd.DataFrame(result.trades)
                        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                        trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")
                        trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
                        
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Trade analysis
                        profit_trades = [float(t['profit'].replace('$', '')) for t in result.trades if float(t['profit'].replace('$', '')) > 0]
                        loss_trades = [float(t['profit'].replace('$', '')) for t in result.trades if float(t['profit'].replace('$', '')) < 0]
                        
                        if profit_trades and loss_trades:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Winning Trades:**")
                                st.write(f"Count: {len(profit_trades)}")
                                st.write(f"Average: ${np.mean(profit_trades):.2f}")
                                st.write(f"Best: ${max(profit_trades):.2f}")
                                
                            with col2:
                                st.write("**Losing Trades:**")
                                st.write(f"Count: {len(loss_trades)}")
                                st.write(f"Average: ${np.mean(loss_trades):.2f}")
                                st.write(f"Worst: ${min(loss_trades):.2f}")
                    
                    # Strategy insights
                    st.write("### 💡 Strategy Insights")
                    
                    if result.total_return > 0:
                        st.success(f"✅ The {strategy_name} strategy outperformed with a {result.total_return:.2f}% total return")
                    else:
                        st.error(f"❌ The {strategy_name} strategy underperformed with a {result.total_return:.2f}% total return")
                    
                    if result.sharpe_ratio > 1:
                        st.info(f"📊 Good risk-adjusted returns with Sharpe ratio of {result.sharpe_ratio:.2f}")
                    elif result.sharpe_ratio > 0:
                        st.warning(f"⚠️ Moderate risk-adjusted returns with Sharpe ratio of {result.sharpe_ratio:.2f}")
                    else:
                        st.error(f"🔻 Poor risk-adjusted returns with Sharpe ratio of {result.sharpe_ratio:.2f}")
                    
                    if result.max_drawdown < -20:
                        st.error(f"🔴 High maximum drawdown of {result.max_drawdown:.2f}% indicates significant risk")
                    elif result.max_drawdown < -10:
                        st.warning(f"🟡 Moderate maximum drawdown of {result.max_drawdown:.2f}%")
                    else:
                        st.success(f"🟢 Low maximum drawdown of {result.max_drawdown:.2f}% shows good risk control")
                else:
                    st.error("Failed to run backtest. Please check your parameters and try again.")
                    
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                
        # Strategy explanations
        st.write("### 📚 Strategy Explanations")
        
        if strategy_name == "SMA Crossover":
            st.info("""
            **Simple Moving Average Crossover:**
            - Buys when fast SMA crosses above slow SMA
            - Sells when fast SMA crosses below slow SMA
            - Works well in trending markets
            - Can produce false signals in sideways markets
            """)
            
        elif strategy_name == "RSI Mean Reversion":
            st.info("""
            **RSI Mean Reversion:**
            - Buys when RSI falls below oversold level (default 30)
            - Sells when RSI rises above overbought level (default 70)
            - Assumes prices will revert to the mean
            - Works well in range-bound markets
            """)
            
        elif strategy_name == "Bollinger Bands":
            st.info("""
            **Bollinger Bands Strategy:**
            - Buys when price touches lower band
            - Sells when price touches upper band
            - Based on mean reversion concept
            - Bands adjust to market volatility
            """)
    
    # Quick strategy comparison
    st.write("### 🔬 Quick Strategy Comparison")
    
    if st.button("Compare All Strategies"):
        try:
            with st.spinner("Comparing strategies..."):
                backtester = SimpleBacktester(initial_capital=initial_capital)
                
                strategies = [
                    ("SMA Crossover", sma_crossover_strategy, {'fast_period': 10, 'slow_period': 20}),
                    ("RSI Mean Reversion", rsi_strategy, {'rsi_period': 14, 'oversold': 30, 'overbought': 70}),
                    ("Bollinger Bands", bollinger_bands_strategy, {'period': 20, 'num_std': 2.0})
                ]
                
                comparison_results = []
                
                for name, func, params in strategies:
                    result = backtester.run_backtest(
                        strategy_func=func,
                        symbol=symbol.upper(),
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        **params
                    )
                    
                    if result:
                        comparison_results.append({
                            'Strategy': name,
                            'Total Return (%)': f"{result.total_return:.2f}",
                            'Annual Return (%)': f"{result.annual_return:.2f}",
                            'Max Drawdown (%)': f"{result.max_drawdown:.2f}",
                            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                            'Win Rate (%)': f"{result.win_rate:.1f}",
                            'Trades': result.num_trades
                        })
                
                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Find best strategy
                    returns = [float(r['Total Return (%)']) for r in comparison_results]
                    best_idx = returns.index(max(returns))
                    best_strategy = comparison_results[best_idx]['Strategy']
                    
                    st.success(f"🏆 Best performing strategy: **{best_strategy}** with {max(returns):.2f}% return")
                    
        except Exception as e:
            st.error(f"Error comparing strategies: {e}")
            
    # Footer tips
    st.markdown("---")
    st.markdown("""
    **💡 Backtesting Tips:**
    - Past performance doesn't guarantee future results
    - Test multiple time periods and market conditions
    - Consider transaction costs and slippage in real trading
    - Combine technical analysis with fundamental analysis
    - Use proper position sizing and risk management
    """)

def render_backtesting(session_state):
    """Render the main backtesting page"""
    render_simple_backtesting()