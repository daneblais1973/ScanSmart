"""
Simple backtesting framework for trading strategies
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    num_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Dict]
    equity_curve: pd.Series

class SimpleBacktester:
    """Simple backtesting framework for strategy validation"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, 
                    strategy_func: Callable,
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    **strategy_params) -> BacktestResult:
        """Run a backtest for a given strategy"""
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Initialize tracking variables
            position = 0  # 0 = no position, 1 = long, -1 = short
            cash = self.initial_capital
            equity = self.initial_capital
            trades = []
            equity_curve = []
            
            # Run strategy on each day
            for i in range(len(data)):
                current_data = data.iloc[:i+1]  # Data up to current point
                current_price = data.iloc[i]['Close']
                
                if len(current_data) < 20:  # Need minimum data for indicators
                    equity_curve.append(equity)
                    continue
                
                # Get strategy signal
                signal = strategy_func(current_data, **strategy_params)
                
                # Execute trades based on signal
                if signal == 'buy' and position <= 0:
                    # Close short position if any
                    if position == -1:
                        profit = (entry_price - current_price) * shares
                        cash += profit
                        trades.append({
                            'date': data.index[i],
                            'type': 'cover',
                            'price': current_price,
                            'shares': shares,
                            'profit': profit
                        })
                    
                    # Open long position
                    shares = int(cash * 0.95 / current_price)  # Use 95% of cash
                    if shares > 0:
                        position = 1
                        entry_price = current_price
                        cash -= shares * current_price
                        trades.append({
                            'date': data.index[i],
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'profit': 0
                        })
                
                elif signal == 'sell' and position >= 0:
                    # Close long position if any
                    if position == 1:
                        profit = (current_price - entry_price) * shares
                        cash += shares * current_price
                        trades.append({
                            'date': data.index[i],
                            'type': 'sell',
                            'price': current_price,
                            'shares': shares,
                            'profit': profit
                        })
                    
                    # Open short position (simplified - not all brokers allow)
                    shares = int(cash * 0.95 / current_price)
                    if shares > 0:
                        position = -1
                        entry_price = current_price
                        cash += shares * current_price  # Credit from short sale
                        trades.append({
                            'date': data.index[i],
                            'type': 'short',
                            'price': current_price,
                            'shares': shares,
                            'profit': 0
                        })
                
                # Calculate current equity
                if position == 1:  # Long position
                    equity = cash + shares * current_price
                elif position == -1:  # Short position
                    equity = cash - shares * current_price
                else:  # No position
                    equity = cash
                
                equity_curve.append(equity)
            
            # Close any remaining position
            if position != 0:
                final_price = data.iloc[-1]['Close']
                if position == 1:
                    profit = (final_price - entry_price) * shares
                    cash += shares * final_price
                else:  # position == -1
                    profit = (entry_price - final_price) * shares
                    cash += profit
                
                trades.append({
                    'date': data.index[-1],
                    'type': 'close',
                    'price': final_price,
                    'shares': shares,
                    'profit': profit
                })
            
            # Calculate performance metrics
            equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
            returns = equity_series.pct_change().dropna()
            
            total_return = (equity_series.iloc[-1] / self.initial_capital - 1) * 100
            annual_return = (1 + total_return/100) ** (252 / len(data)) - 1
            
            # Max drawdown
            running_max = equity_series.expanding().max()
            drawdowns = (equity_series - running_max) / running_max
            max_drawdown = drawdowns.min() * 100
            
            # Sharpe ratio
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Trading statistics
            profit_trades = [t for t in trades if t['profit'] > 0]
            loss_trades = [t for t in trades if t['profit'] < 0]
            
            num_trades = len([t for t in trades if t['type'] in ['buy', 'short']])
            win_rate = len(profit_trades) / len(trades) * 100 if trades else 0
            
            total_profit = sum(t['profit'] for t in profit_trades)
            total_loss = abs(sum(t['profit'] for t in loss_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            return BacktestResult(
                strategy_name=strategy_func.__name__,
                total_return=total_return,
                annual_return=annual_return * 100,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                num_trades=num_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades=trades,
                equity_curve=equity_series
            )
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return None

# Sample strategies for testing
def sma_crossover_strategy(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 20) -> str:
    """Simple moving average crossover strategy"""
    if len(data) < slow_period:
        return 'hold'
    
    fast_sma = data['Close'].rolling(fast_period).mean()
    slow_sma = data['Close'].rolling(slow_period).mean()
    
    if fast_sma.iloc[-1] > slow_sma.iloc[-1] and fast_sma.iloc[-2] <= slow_sma.iloc[-2]:
        return 'buy'
    elif fast_sma.iloc[-1] < slow_sma.iloc[-1] and fast_sma.iloc[-2] >= slow_sma.iloc[-2]:
        return 'sell'
    else:
        return 'hold'

def rsi_strategy(data: pd.DataFrame, rsi_period: int = 14, oversold: float = 30, overbought: float = 70) -> str:
    """RSI mean reversion strategy"""
    if len(data) < rsi_period + 1:
        return 'hold'
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    
    if current_rsi < oversold:
        return 'buy'
    elif current_rsi > overbought:
        return 'sell'
    else:
        return 'hold'

def bollinger_bands_strategy(data: pd.DataFrame, period: int = 20, num_std: float = 2) -> str:
    """Bollinger Bands mean reversion strategy"""
    if len(data) < period:
        return 'hold'
    
    sma = data['Close'].rolling(period).mean()
    std = data['Close'].rolling(period).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    current_price = data['Close'].iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    if current_price < current_lower:
        return 'buy'
    elif current_price > current_upper:
        return 'sell'
    else:
        return 'hold'