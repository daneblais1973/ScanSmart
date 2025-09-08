"""
Advanced analytics module for portfolio management, risk analysis, and performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    beta: float
    alpha: float
    var_95: float
    cvar_95: float

@dataclass
class RiskMetrics:
    var_1d: float
    var_5d: float
    expected_shortfall: float
    maximum_drawdown: float
    drawdown_duration: int
    beta_to_market: float
    correlation_to_market: float

class AdvancedAnalytics:
    """Advanced analytics for portfolio and risk management"""
    
    def __init__(self, config=None):
        self.config = config
        
    def monte_carlo_simulation(self, 
                             symbol: str, 
                             num_simulations: int = 1000,
                             time_horizon: int = 252,  # 1 year
                             initial_price: float = None) -> Dict:
        """Run Monte Carlo simulation for stock price prediction"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {}
                
            if initial_price is None:
                initial_price = hist['Close'].iloc[-1]
                
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            simulations = []
            
            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, time_horizon)
                
                # Calculate price path
                price_path = [initial_price]
                for ret in random_returns:
                    price_path.append(price_path[-1] * (1 + ret))
                    
                simulations.append(price_path)
                
            simulations = np.array(simulations)
            
            # Calculate statistics
            final_prices = simulations[:, -1]
            
            results = {
                'symbol': symbol,
                'initial_price': initial_price,
                'simulations': simulations.tolist(),
                'final_prices': final_prices.tolist(),
                'statistics': {
                    'mean_final_price': np.mean(final_prices),
                    'median_final_price': np.median(final_prices),
                    'std_final_price': np.std(final_prices),
                    'percentile_5': np.percentile(final_prices, 5),
                    'percentile_25': np.percentile(final_prices, 25),
                    'percentile_75': np.percentile(final_prices, 75),
                    'percentile_95': np.percentile(final_prices, 95),
                    'probability_positive': (final_prices > initial_price).mean(),
                    'expected_return': (np.mean(final_prices) / initial_price - 1) * 100
                },
                'time_horizon_days': time_horizon,
                'num_simulations': num_simulations
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation for {symbol}: {e}")
            return {}
            
    def calculate_portfolio_metrics(self, 
                                  returns: pd.Series, 
                                  benchmark_returns: pd.Series = None,
                                  risk_free_rate: float = 0.02) -> PortfolioMetrics:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Basic metrics
            total_return = (1 + returns).cumprod().iloc[-1] - 1
            annualized_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            # Sortino ratio (downside deviation only)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Beta and Alpha (if benchmark provided)
            beta = alpha = 0
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            return PortfolioMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                beta=beta,
                alpha=alpha,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return None
            
    def calculate_position_sizing(self, 
                                portfolio_value: float,
                                risk_per_trade: float = 0.02,  # 2% risk per trade
                                entry_price: float = None,
                                stop_loss_price: float = None,
                                method: str = "kelly") -> Dict:
        """Calculate optimal position sizing"""
        try:
            if method == "fixed_percent":
                position_size = portfolio_value * risk_per_trade
                return {
                    'method': 'Fixed Percentage',
                    'position_value': position_size,
                    'risk_amount': position_size,
                    'percentage_of_portfolio': risk_per_trade * 100
                }
                
            elif method == "kelly" and entry_price and stop_loss_price:
                # Kelly Criterion (simplified)
                risk_per_share = abs(entry_price - stop_loss_price)
                max_risk_amount = portfolio_value * risk_per_trade
                shares = int(max_risk_amount / risk_per_share)
                position_value = shares * entry_price
                
                return {
                    'method': 'Kelly Criterion',
                    'shares': shares,
                    'position_value': position_value,
                    'risk_per_share': risk_per_share,
                    'total_risk': shares * risk_per_share,
                    'percentage_of_portfolio': (position_value / portfolio_value) * 100
                }
                
            elif method == "volatility_adjusted":
                # Adjust position size based on volatility
                # This is a simplified version
                base_position = portfolio_value * risk_per_trade
                return {
                    'method': 'Volatility Adjusted',
                    'position_value': base_position,
                    'risk_amount': base_position,
                    'percentage_of_portfolio': risk_per_trade * 100
                }
                
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {}
            
    def calculate_optimal_stop_loss(self, 
                                  symbol: str,
                                  entry_price: float,
                                  method: str = "atr",
                                  period: int = 14) -> Dict:
        """Calculate optimal stop-loss levels"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            
            if hist.empty:
                return {}
                
            if method == "atr":
                # Average True Range method
                high_low = hist['High'] - hist['Low']
                high_close = np.abs(hist['High'] - hist['Close'].shift())
                low_close = np.abs(hist['Low'] - hist['Close'].shift())
                
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(window=period).mean().iloc[-1]
                
                # Different ATR multipliers for different risk levels
                stop_levels = {
                    'conservative': entry_price - (atr * 1.5),
                    'moderate': entry_price - (atr * 2.0),
                    'aggressive': entry_price - (atr * 2.5)
                }
                
                return {
                    'method': 'ATR',
                    'atr_value': atr,
                    'stop_levels': stop_levels,
                    'recommended': stop_levels['moderate']
                }
                
            elif method == "support_resistance":
                # Support/Resistance method (simplified)
                lows = hist['Low'].rolling(window=5).min()
                support_level = lows.iloc[-10:].min()  # Lowest point in last 10 periods
                
                return {
                    'method': 'Support/Resistance',
                    'support_level': support_level,
                    'recommended': support_level * 0.98  # Slightly below support
                }
                
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}")
            return {}
            
    def create_correlation_matrix(self, symbols: List[str], period: str = "1y") -> Dict:
        """Create correlation matrix for given symbols"""
        try:
            # Fetch data for all symbols
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close'].pct_change().dropna()
                    
            if not data:
                return {}
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'symbols': symbols,
                'period': period,
                'matrix_values': correlation_matrix.values.tolist(),
                'strongest_correlation': {
                    'pairs': [],
                    'values': []
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return {}
            
    def analyze_drawdowns(self, returns: pd.Series) -> Dict:
        """Analyze drawdown characteristics"""
        try:
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cumulative - running_max) / running_max
            
            # Find drawdown periods
            in_drawdown = drawdowns < -0.01  # 1% threshold
            
            # Calculate statistics
            max_drawdown = drawdowns.min()
            avg_drawdown = drawdowns[drawdowns < 0].mean()
            
            # Find longest drawdown period
            drawdown_periods = []
            current_period = 0
            
            for dd in in_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
                    
            if current_period > 0:
                drawdown_periods.append(current_period)
                
            longest_drawdown = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_period = np.mean(drawdown_periods) if drawdown_periods else 0
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'longest_drawdown_days': longest_drawdown,
                'avg_drawdown_period': avg_drawdown_period,
                'num_drawdown_periods': len(drawdown_periods),
                'drawdown_series': drawdowns.tolist(),
                'dates': drawdowns.index.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {e}")
            return {}
            
    def calculate_risk_metrics(self, returns: pd.Series, confidence_level: float = 0.95) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # VaR calculations
            var_1d = np.percentile(returns, (1 - confidence_level) * 100)
            var_5d = np.percentile(returns, (1 - confidence_level) * 100) * np.sqrt(5)
            
            # Expected Shortfall (CVaR)
            expected_shortfall = returns[returns <= var_1d].mean()
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            maximum_drawdown = drawdowns.min()
            
            # Drawdown duration
            in_drawdown = drawdowns < -0.01
            drawdown_periods = []
            current_period = 0
            
            for dd in in_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
                    
            drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Market correlation (using S&P 500 as proxy)
            try:
                spy = yf.Ticker('^GSPC')
                spy_hist = spy.history(period="1y")
                spy_returns = spy_hist['Close'].pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(spy_returns.index)
                if len(common_dates) > 30:  # Need sufficient data
                    portfolio_aligned = returns.loc[common_dates]
                    market_aligned = spy_returns.loc[common_dates]
                    
                    correlation_to_market = portfolio_aligned.corr(market_aligned)
                    
                    # Beta calculation
                    covariance = np.cov(portfolio_aligned, market_aligned)[0][1]
                    market_variance = np.var(market_aligned)
                    beta_to_market = covariance / market_variance if market_variance > 0 else 0
                else:
                    correlation_to_market = 0
                    beta_to_market = 0
                    
            except:
                correlation_to_market = 0
                beta_to_market = 0
            
            return RiskMetrics(
                var_1d=var_1d,
                var_5d=var_5d,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                drawdown_duration=drawdown_duration,
                beta_to_market=beta_to_market,
                correlation_to_market=correlation_to_market
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None
            
    def volatility_forecasting(self, symbol: str, days_ahead: int = 30) -> Dict:
        """Forecast volatility using GARCH-like approach"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {}
                
            returns = hist['Close'].pct_change().dropna()
            
            # Simple volatility forecasting using exponential smoothing
            # In production, you'd use proper GARCH models
            
            # Calculate historical volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            
            # Simple exponential smoothing for forecasting
            alpha = 0.1  # Smoothing parameter
            current_vol = rolling_vol.iloc[-1]
            
            # Forecast assuming mean reversion
            long_term_vol = rolling_vol.mean()
            forecast_vol = []
            
            for day in range(days_ahead):
                # Simple mean reversion model
                next_vol = current_vol + alpha * (long_term_vol - current_vol)
                forecast_vol.append(next_vol)
                current_vol = next_vol
                
            return {
                'symbol': symbol,
                'current_volatility': rolling_vol.iloc[-1],
                'historical_volatility': rolling_vol.tolist(),
                'forecast_volatility': forecast_vol,
                'long_term_average': long_term_vol,
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility for {symbol}: {e}")
            return {}