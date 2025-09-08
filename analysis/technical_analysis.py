import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading

from shared.models import Catalyst, CatalystType
from core.config import AppConfig

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Advanced technical analysis for trading signal generation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Comprehensive technical indicator parameters
        self.indicators_config = {
            # Trend Indicators
            'sma_periods': [9, 20, 50, 200],
            'ema_periods': [12, 26, 50],
            'wma_periods': [10, 20],
            'hull_ma_period': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'parabolic_sar_step': 0.02,
            'parabolic_sar_max': 0.2,
            'supertrend_period': 14,
            'supertrend_multiplier': 3,
            'tema_period': 14,
            'dema_period': 14,
            
            # Momentum Indicators
            'rsi_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'williams_r_period': 14,
            'momentum_period': 12,
            'roc_period': 12,
            'trix_period': 14,
            'ultimate_osc_short': 7,
            'ultimate_osc_medium': 14,
            'ultimate_osc_long': 28,
            
            # Volatility Indicators
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'keltner_period': 20,
            'keltner_multiplier': 2,
            'donchian_period': 20,
            'chaikin_vol_period': 14,
            
            # Volume Indicators
            'volume_sma': 20,
            'obv_period': 20,
            'cmf_period': 20,
            'mfi_period': 14,
            'eom_period': 14,
            'vwap_period': 20,
            
            # Oscillators
            'ppo_fast': 12,
            'ppo_slow': 26,
            'ppo_signal': 9,
            'cci_period': 20,
            'dpo_period': 20,
            'fisher_period': 10,
            'schaff_period': 23,
            
            # Pattern Recognition
            'pivot_lookback': 5,
            'fibonacci_lookback': 50,
            'pattern_lookback': 20
        }
        
        # Trading pattern weights for different timeframes
        self.pattern_weights = {
            'day_trading': {
                'momentum': 0.4,
                'volume': 0.3,
                'volatility': 0.2,
                'support_resistance': 0.1
            },
            'momentum_trading': {
                'momentum': 0.3,
                'trend': 0.3,
                'volume': 0.2,
                'breakout': 0.2
            },
            'long_term': {
                'trend': 0.4,
                'support_resistance': 0.3,
                'momentum': 0.2,
                'fundamentals': 0.1
            }
        }
        
        # Dynamic parameter optimization settings
        self.adaptive_parameters = getattr(config.fetchers, 'dynamic_parameters', True)
        self.volatility_adjustments = {
            'low': {'rsi_period': 21, 'bb_std': 1.5, 'atr_multiplier': 0.8},
            'medium': {'rsi_period': 14, 'bb_std': 2.0, 'atr_multiplier': 1.0},
            'high': {'rsi_period': 10, 'bb_std': 2.5, 'atr_multiplier': 1.3},
            'extreme': {'rsi_period': 7, 'bb_std': 3.0, 'atr_multiplier': 1.6}
        }
        
        logger.info("Technical Analyzer initialized")
    
    async def analyze_stock_technical(self, ticker: str, period: str = '1mo', 
                                    interval: str = '1d', catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Comprehensive technical analysis for a stock"""
        try:
            # Get stock data
            stock_data = await self._get_stock_data(ticker, period, interval)
            if stock_data is None or stock_data.empty:
                logger.warning(f"No stock data available for {ticker}")
                return {}
            
            # Apply dynamic parameter optimization
            if self.adaptive_parameters:
                self._optimize_parameters_for_market_conditions(stock_data)
            
            # Calculate all technical indicators
            indicators = await self._calculate_all_indicators(stock_data)
            
            # Analyze patterns
            patterns = await self._detect_patterns(stock_data, indicators)
            
            # Calculate signals for different trading types
            trading_signals = await self._generate_trading_signals(
                stock_data, indicators, patterns, catalyst
            )
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(stock_data)
            
            # Determine overall technical score
            technical_score = await self._calculate_technical_score(
                indicators, patterns, trading_signals
            )
            
            return {
                'ticker': ticker,
                'last_price': float(stock_data['Close'].iloc[-1]),
                'volume': int(stock_data['Volume'].iloc[-1]),
                'indicators': indicators,
                'patterns': patterns,
                'trading_signals': trading_signals,
                'risk_metrics': risk_metrics,
                'technical_score': technical_score,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(stock_data)
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {ticker}: {e}")
            return {}
    
    async def _get_stock_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Get stock price data with caching"""
        cache_key = f"{ticker}_{period}_{interval}"
        
        with self._cache_lock:
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                # Use cached data if less than 5 minutes old
                if datetime.now() - timestamp < timedelta(minutes=5):
                    return cached_data
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(
                self._executor, 
                lambda: yf.Ticker(ticker)
            )
            
            data = await loop.run_in_executor(
                self._executor,
                lambda: stock.history(period=period, interval=interval)
            )
            
            if data is not None and not data.empty:
                with self._cache_lock:
                    self._cache[cache_key] = (data, datetime.now())
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    async def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # Comprehensive technical indicators
            indicators.update(await self._calculate_trend_indicators_comprehensive(data))
            indicators.update(await self._calculate_momentum_indicators_comprehensive(data))
            indicators.update(await self._calculate_volatility_indicators_comprehensive(data))
            indicators.update(await self._calculate_volume_indicators_comprehensive(data))
            indicators.update(await self._calculate_oscillators(data))
            indicators.update(await self._calculate_pattern_indicators(data))
            
            # Legacy methods for backward compatibility
            indicators.update(await self._calculate_moving_averages(data))
            indicators.update(await self._calculate_momentum_indicators(data))
            indicators.update(await self._calculate_volatility_indicators(data))
            indicators.update(await self._calculate_volume_indicators(data))
            indicators.update(await self._calculate_trend_indicators(data))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    async def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving average indicators"""
        try:
            ma_data = {}
            
            # Simple Moving Averages
            for period in self.indicators_config['sma_periods']:
                if len(data) >= period:
                    ma_data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean().iloc[-1]
            
            # Exponential Moving Averages
            for period in self.indicators_config['ema_periods']:
                if len(data) >= period:
                    ma_data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean().iloc[-1]
            
            # Current price relative to moving averages
            current_price = data['Close'].iloc[-1]
            
            ma_signals = []
            if 'SMA_20' in ma_data:
                ma_signals.append(1 if current_price > ma_data['SMA_20'] else -1)
            if 'SMA_50' in ma_data:
                ma_signals.append(1 if current_price > ma_data['SMA_50'] else -1)
            
            ma_data['ma_signal_strength'] = sum(ma_signals) / len(ma_signals) if ma_signals else 0
            
            return ma_data
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    async def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        try:
            momentum_data = {}
            
            # RSI
            if len(data) >= self.indicators_config['rsi_period'] + 1:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.indicators_config['rsi_period']).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicators_config['rsi_period']).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                momentum_data['RSI'] = rsi.iloc[-1]
                
                # RSI signals
                if rsi.iloc[-1] > 70:
                    momentum_data['RSI_signal'] = -1  # Overbought
                elif rsi.iloc[-1] < 30:
                    momentum_data['RSI_signal'] = 1   # Oversold
                else:
                    momentum_data['RSI_signal'] = 0
            
            # MACD
            if len(data) >= max(self.indicators_config['macd_fast'], self.indicators_config['macd_slow']):
                ema_fast = data['Close'].ewm(span=self.indicators_config['macd_fast']).mean()
                ema_slow = data['Close'].ewm(span=self.indicators_config['macd_slow']).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=self.indicators_config['macd_signal']).mean()
                histogram = macd_line - signal_line
                
                momentum_data['MACD'] = macd_line.iloc[-1]
                momentum_data['MACD_signal'] = signal_line.iloc[-1]
                momentum_data['MACD_histogram'] = histogram.iloc[-1]
                
                # MACD crossover signal
                if len(histogram) > 1:
                    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                        momentum_data['MACD_crossover'] = 1  # Bullish crossover
                    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                        momentum_data['MACD_crossover'] = -1  # Bearish crossover
                    else:
                        momentum_data['MACD_crossover'] = 0
            
            # Stochastic Oscillator
            if len(data) >= self.indicators_config['stoch_k']:
                low_min = data['Low'].rolling(window=self.indicators_config['stoch_k']).min()
                high_max = data['High'].rolling(window=self.indicators_config['stoch_k']).max()
                k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
                d_percent = k_percent.rolling(window=self.indicators_config['stoch_d']).mean()
                
                momentum_data['Stoch_K'] = k_percent.iloc[-1]
                momentum_data['Stoch_D'] = d_percent.iloc[-1]
                
                # Stochastic signals
                if k_percent.iloc[-1] > 80:
                    momentum_data['Stoch_signal'] = -1  # Overbought
                elif k_percent.iloc[-1] < 20:
                    momentum_data['Stoch_signal'] = 1   # Oversold
                else:
                    momentum_data['Stoch_signal'] = 0
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    async def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        try:
            volatility_data = {}
            
            # Bollinger Bands
            if len(data) >= self.indicators_config['bb_period']:
                sma = data['Close'].rolling(window=self.indicators_config['bb_period']).mean()
                std = data['Close'].rolling(window=self.indicators_config['bb_period']).std()
                
                upper_band = sma + (std * self.indicators_config['bb_std'])
                lower_band = sma - (std * self.indicators_config['bb_std'])
                
                volatility_data['BB_upper'] = upper_band.iloc[-1]
                volatility_data['BB_middle'] = sma.iloc[-1]
                volatility_data['BB_lower'] = lower_band.iloc[-1]
                
                # Bollinger Band position
                current_price = data['Close'].iloc[-1]
                bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
                volatility_data['BB_position'] = bb_position
                
                # BB signals
                if current_price > upper_band.iloc[-1]:
                    volatility_data['BB_signal'] = -1  # Price above upper band
                elif current_price < lower_band.iloc[-1]:
                    volatility_data['BB_signal'] = 1   # Price below lower band
                else:
                    volatility_data['BB_signal'] = 0
            
            # Average True Range (ATR)
            if len(data) >= 14:
                high_low = data['High'] - data['Low']
                high_close_prev = np.abs(data['High'] - data['Close'].shift())
                low_close_prev = np.abs(data['Low'] - data['Close'].shift())
                
                tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
                atr = tr.rolling(window=14).mean()
                volatility_data['ATR'] = atr.iloc[-1]
                
                # Volatility percentile (current ATR vs recent ATR values)
                if len(atr) >= 50:
                    recent_atr = atr.iloc[-50:]
                    volatility_data['volatility_percentile'] = (atr.iloc[-1] > recent_atr.quantile(0.75))
            
            return volatility_data
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    async def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            volume_data = {}
            
            # Volume Moving Average
            if len(data) >= self.indicators_config['volume_sma']:
                volume_sma = data['Volume'].rolling(window=self.indicators_config['volume_sma']).mean()
                volume_data['Volume_SMA'] = volume_sma.iloc[-1]
                
                # Volume ratio (current vs average)
                current_volume = data['Volume'].iloc[-1]
                volume_data['volume_ratio'] = current_volume / volume_sma.iloc[-1]
                
                # Unusual volume detection
                volume_data['unusual_volume'] = volume_data['volume_ratio'] > 2.0
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            volume_data['OBV'] = obv[-1]
            
            # OBV trend
            if len(obv) >= 10:
                recent_obv = obv[-10:]
                obv_trend = (recent_obv[-1] - recent_obv[0]) / len(recent_obv)
                volume_data['OBV_trend'] = obv_trend
            
            return volume_data
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    async def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        try:
            trend_data = {}
            
            # Price trend (linear regression slope)
            if len(data) >= 20:
                prices = data['Close'].iloc[-20:].values
                x = np.arange(len(prices))
                slope, _ = np.polyfit(x, prices, 1)
                trend_data['price_trend_slope'] = slope
                
                # Trend strength
                correlation = np.corrcoef(x, prices)[0, 1]
                trend_data['trend_strength'] = abs(correlation)
                
                # Trend direction
                if slope > 0 and correlation > 0.7:
                    trend_data['trend_direction'] = 1   # Strong uptrend
                elif slope < 0 and correlation < -0.7:
                    trend_data['trend_direction'] = -1  # Strong downtrend
                else:
                    trend_data['trend_direction'] = 0   # Sideways/weak trend
            
            # Support and Resistance Levels
            if len(data) >= 50:
                highs = data['High'].iloc[-50:]
                lows = data['Low'].iloc[-50:]
                
                # Find recent highs and lows
                resistance_level = highs.quantile(0.95)
                support_level = lows.quantile(0.05)
                
                trend_data['resistance_level'] = resistance_level
                trend_data['support_level'] = support_level
                
                current_price = data['Close'].iloc[-1]
                
                # Distance to support/resistance
                trend_data['distance_to_resistance'] = (resistance_level - current_price) / current_price
                trend_data['distance_to_support'] = (current_price - support_level) / current_price
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {}
    
    async def _detect_patterns(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Detect chart patterns and formations"""
        try:
            patterns = {}
            
            # Breakout patterns
            patterns.update(await self._detect_breakout_patterns(data, indicators))
            
            # Reversal patterns
            patterns.update(await self._detect_reversal_patterns(data, indicators))
            
            # Continuation patterns
            patterns.update(await self._detect_continuation_patterns(data, indicators))
            
            # Volume patterns
            patterns.update(await self._detect_volume_patterns(data, indicators))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {}
    
    async def _detect_breakout_patterns(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Detect breakout patterns"""
        try:
            breakout_patterns = {}
            
            current_price = data['Close'].iloc[-1]
            
            # Bollinger Band breakout
            if 'BB_upper' in indicators and 'BB_lower' in indicators:
                if current_price > indicators['BB_upper']:
                    breakout_patterns['bb_breakout'] = 'bullish'
                elif current_price < indicators['BB_lower']:
                    breakout_patterns['bb_breakout'] = 'bearish'
                else:
                    breakout_patterns['bb_breakout'] = None
            
            # Resistance/Support breakout
            if 'resistance_level' in indicators and 'support_level' in indicators:
                if current_price > indicators['resistance_level']:
                    breakout_patterns['resistance_breakout'] = True
                elif current_price < indicators['support_level']:
                    breakout_patterns['support_breakdown'] = True
            
            # Volume confirmation for breakouts
            if 'volume_ratio' in indicators and indicators['volume_ratio'] > 1.5:
                breakout_patterns['volume_confirmed'] = True
            
            return breakout_patterns
            
        except Exception as e:
            logger.error(f"Error detecting breakout patterns: {e}")
            return {}
    
    async def _detect_reversal_patterns(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Detect reversal patterns"""
        try:
            reversal_patterns = {}
            
            # RSI divergence
            if 'RSI' in indicators:
                if indicators['RSI'] < 30:
                    reversal_patterns['oversold_reversal_signal'] = 'bullish'
                elif indicators['RSI'] > 70:
                    reversal_patterns['overbought_reversal_signal'] = 'bearish'
            
            # MACD reversal
            if 'MACD_crossover' in indicators:
                if indicators['MACD_crossover'] == 1:
                    reversal_patterns['macd_bullish_reversal'] = True
                elif indicators['MACD_crossover'] == -1:
                    reversal_patterns['macd_bearish_reversal'] = True
            
            # Double top/bottom detection (simplified)
            if len(data) >= 30:
                recent_highs = data['High'].iloc[-30:]
                recent_lows = data['Low'].iloc[-30:]
                
                # Look for double top pattern
                max_high = recent_highs.max()
                high_indices = recent_highs[recent_highs > max_high * 0.98].index
                if len(high_indices) >= 2:
                    reversal_patterns['potential_double_top'] = True
                
                # Look for double bottom pattern
                min_low = recent_lows.min()
                low_indices = recent_lows[recent_lows < min_low * 1.02].index
                if len(low_indices) >= 2:
                    reversal_patterns['potential_double_bottom'] = True
            
            return reversal_patterns
            
        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {e}")
            return {}
    
    async def _detect_continuation_patterns(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Detect continuation patterns"""
        try:
            continuation_patterns = {}
            
            # Moving average alignment
            ma_alignment_score = 0
            current_price = data['Close'].iloc[-1]
            
            ma_keys = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
            valid_mas = [indicators[key] for key in ma_keys if key in indicators]
            
            if valid_mas:
                bullish_alignment = all(current_price > ma for ma in valid_mas)
                bearish_alignment = all(current_price < ma for ma in valid_mas)
                
                if bullish_alignment:
                    continuation_patterns['ma_alignment'] = 'bullish'
                elif bearish_alignment:
                    continuation_patterns['ma_alignment'] = 'bearish'
                else:
                    continuation_patterns['ma_alignment'] = 'mixed'
            
            # Trend continuation
            if 'trend_direction' in indicators and 'trend_strength' in indicators:
                if indicators['trend_strength'] > 0.7:
                    if indicators['trend_direction'] == 1:
                        continuation_patterns['strong_uptrend'] = True
                    elif indicators['trend_direction'] == -1:
                        continuation_patterns['strong_downtrend'] = True
            
            return continuation_patterns
            
        except Exception as e:
            logger.error(f"Error detecting continuation patterns: {e}")
            return {}
    
    async def _detect_volume_patterns(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Detect volume-based patterns"""
        try:
            volume_patterns = {}
            
            # Volume spike
            if 'volume_ratio' in indicators:
                if indicators['volume_ratio'] > 3.0:
                    volume_patterns['extreme_volume_spike'] = True
                elif indicators['volume_ratio'] > 2.0:
                    volume_patterns['volume_spike'] = True
            
            # Volume trend
            if len(data) >= 10:
                recent_volumes = data['Volume'].iloc[-10:].values
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                
                if volume_trend > 0:
                    volume_patterns['increasing_volume_trend'] = True
                else:
                    volume_patterns['decreasing_volume_trend'] = True
            
            # OBV confirmation
            if 'OBV_trend' in indicators:
                if indicators['OBV_trend'] > 0:
                    volume_patterns['obv_bullish'] = True
                else:
                    volume_patterns['obv_bearish'] = True
            
            return volume_patterns
            
        except Exception as e:
            logger.error(f"Error detecting volume patterns: {e}")
            return {}
    
    async def _generate_trading_signals(self, data: pd.DataFrame, indicators: Dict, 
                                      patterns: Dict, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate trading signals for different trading styles"""
        try:
            trading_signals = {}
            
            # Day trading signals (focus on momentum and volatility)
            trading_signals['day_trading'] = await self._generate_day_trading_signals(
                data, indicators, patterns, catalyst
            )
            
            # Momentum/Swing trading signals
            trading_signals['momentum_trading'] = await self._generate_momentum_signals(
                data, indicators, patterns, catalyst
            )
            
            # Long-term signals
            trading_signals['long_term'] = await self._generate_long_term_signals(
                data, indicators, patterns, catalyst
            )
            
            return trading_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {}
    
    async def _generate_day_trading_signals(self, data: pd.DataFrame, indicators: Dict, 
                                          patterns: Dict, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate day trading specific signals"""
        try:
            signals = {
                'signal_strength': 0.0,
                'direction': 0,  # -1 bearish, 0 neutral, 1 bullish
                'confidence': 0.0,
                'entry_reason': [],
                'risk_level': 'MEDIUM'
            }
            
            score_components = []
            confidence_factors = []
            
            # Momentum factors (high weight for day trading)
            if 'RSI' in indicators:
                rsi = indicators['RSI']
                if 30 <= rsi <= 70:  # Good range for day trading
                    score_components.append(0.3)
                    confidence_factors.append(0.7)
                    if rsi < 40:
                        signals['direction'] = 1
                        signals['entry_reason'].append('RSI oversold bounce opportunity')
                    elif rsi > 60:
                        signals['direction'] = -1
                        signals['entry_reason'].append('RSI overbought pullback opportunity')
            
            # MACD momentum
            if 'MACD_crossover' in indicators and indicators['MACD_crossover'] != 0:
                score_components.append(0.4)
                confidence_factors.append(0.8)
                signals['direction'] = indicators['MACD_crossover']
                signals['entry_reason'].append('MACD crossover signal')
            
            # Volume confirmation (critical for day trading)
            if patterns.get('volume_spike'):
                score_components.append(0.3)
                confidence_factors.append(0.9)
                signals['entry_reason'].append('Volume spike confirmation')
            
            # Volatility (good for day trading)
            if indicators.get('volatility_percentile'):
                score_components.append(0.2)
                confidence_factors.append(0.6)
                signals['entry_reason'].append('High volatility environment')
                signals['risk_level'] = 'HIGH'
            
            # Breakout patterns
            if patterns.get('bb_breakout'):
                score_components.append(0.4)
                confidence_factors.append(0.7)
                direction = 1 if patterns['bb_breakout'] == 'bullish' else -1
                if signals['direction'] == 0:
                    signals['direction'] = direction
                signals['entry_reason'].append(f'Bollinger Band {patterns["bb_breakout"]} breakout')
            
            # Catalyst boost
            if catalyst and catalyst.impact >= 70:
                catalyst_boost = catalyst.impact / 100.0 * 0.3
                score_components.append(catalyst_boost)
                confidence_factors.append(0.8)
                catalyst_direction = 1 if catalyst.sentiment_score > 0 else -1
                if signals['direction'] == 0:
                    signals['direction'] = catalyst_direction
                signals['entry_reason'].append('High-impact catalyst detected')
            
            # Calculate final scores
            if score_components:
                signals['signal_strength'] = min(1.0, sum(score_components))
                signals['confidence'] = min(1.0, sum(confidence_factors) / len(confidence_factors))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating day trading signals: {e}")
            return {'signal_strength': 0.0, 'direction': 0, 'confidence': 0.0, 'entry_reason': [], 'risk_level': 'HIGH'}
    
    async def _generate_momentum_signals(self, data: pd.DataFrame, indicators: Dict, 
                                       patterns: Dict, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate momentum/swing trading signals"""
        try:
            signals = {
                'signal_strength': 0.0,
                'direction': 0,
                'confidence': 0.0,
                'entry_reason': [],
                'risk_level': 'MEDIUM'
            }
            
            score_components = []
            confidence_factors = []
            
            # Trend strength (key for momentum trading)
            if 'trend_strength' in indicators and 'trend_direction' in indicators:
                if indicators['trend_strength'] > 0.6:
                    score_components.append(0.4)
                    confidence_factors.append(0.9)
                    signals['direction'] = indicators['trend_direction']
                    trend_type = 'uptrend' if indicators['trend_direction'] == 1 else 'downtrend'
                    signals['entry_reason'].append(f'Strong {trend_type} detected')
            
            # Moving average alignment
            if patterns.get('ma_alignment') in ['bullish', 'bearish']:
                score_components.append(0.3)
                confidence_factors.append(0.8)
                direction = 1 if patterns['ma_alignment'] == 'bullish' else -1
                if signals['direction'] == 0:
                    signals['direction'] = direction
                signals['entry_reason'].append(f'Moving average {patterns["ma_alignment"]} alignment')
            
            # Momentum oscillators
            momentum_score = 0
            if 'RSI' in indicators:
                rsi = indicators['RSI']
                if 40 <= rsi <= 60:  # Good momentum range
                    momentum_score += 0.3
            
            if momentum_score > 0:
                score_components.append(momentum_score)
                confidence_factors.append(0.6)
                signals['entry_reason'].append('Favorable momentum conditions')
            
            # Volume trend support
            if patterns.get('increasing_volume_trend'):
                score_components.append(0.2)
                confidence_factors.append(0.7)
                signals['entry_reason'].append('Increasing volume trend')
            
            # Breakout with volume
            if patterns.get('resistance_breakout') and patterns.get('volume_confirmed'):
                score_components.append(0.5)
                confidence_factors.append(0.9)
                signals['direction'] = 1
                signals['entry_reason'].append('Volume-confirmed resistance breakout')
            
            # Calculate final scores
            if score_components:
                signals['signal_strength'] = min(1.0, sum(score_components))
                signals['confidence'] = min(1.0, sum(confidence_factors) / len(confidence_factors))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {'signal_strength': 0.0, 'direction': 0, 'confidence': 0.0, 'entry_reason': [], 'risk_level': 'MEDIUM'}
    
    async def _generate_long_term_signals(self, data: pd.DataFrame, indicators: Dict, 
                                        patterns: Dict, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate long-term investment signals"""
        try:
            signals = {
                'signal_strength': 0.0,
                'direction': 0,
                'confidence': 0.0,
                'entry_reason': [],
                'risk_level': 'LOW'
            }
            
            score_components = []
            confidence_factors = []
            
            # Long-term trend (most important for long-term)
            if 'SMA_200' in indicators:
                current_price = data['Close'].iloc[-1]
                if current_price > indicators['SMA_200']:
                    score_components.append(0.4)
                    confidence_factors.append(0.9)
                    signals['direction'] = 1
                    signals['entry_reason'].append('Above 200-day moving average')
                else:
                    signals['direction'] = -1
                    signals['entry_reason'].append('Below 200-day moving average')
            
            # Support/resistance levels
            if 'distance_to_support' in indicators:
                if indicators['distance_to_support'] < 0.05:  # Near support
                    score_components.append(0.3)
                    confidence_factors.append(0.7)
                    if signals['direction'] == 0:
                        signals['direction'] = 1
                    signals['entry_reason'].append('Near support level - good entry')
            
            # Stable trend
            if patterns.get('strong_uptrend') or patterns.get('strong_downtrend'):
                score_components.append(0.3)
                confidence_factors.append(0.8)
                direction = 1 if patterns.get('strong_uptrend') else -1
                if signals['direction'] == 0:
                    signals['direction'] = direction
                trend_type = 'up' if direction == 1 else 'down'
                signals['entry_reason'].append(f'Strong {trend_type}trend continuation')
            
            # Calculate final scores
            if score_components:
                signals['signal_strength'] = min(1.0, sum(score_components))
                signals['confidence'] = min(1.0, sum(confidence_factors) / len(confidence_factors))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating long-term signals: {e}")
            return {'signal_strength': 0.0, 'direction': 0, 'confidence': 0.0, 'entry_reason': [], 'risk_level': 'LOW'}
    
    async def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            risk_metrics = {}
            
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                
                # Volatility (annualized)
                daily_vol = returns.std()
                risk_metrics['volatility'] = daily_vol * np.sqrt(252)  # Annualized
                
                # Maximum drawdown
                rolling_max = data['Close'].rolling(window=len(data), min_periods=1).max()
                drawdown = (data['Close'] - rolling_max) / rolling_max
                risk_metrics['max_drawdown'] = drawdown.min()
                
                # Sharpe ratio (assuming risk-free rate of 2%)
                risk_free_rate = 0.02
                excess_return = returns.mean() * 252 - risk_free_rate
                if daily_vol > 0:
                    risk_metrics['sharpe_ratio'] = excess_return / (daily_vol * np.sqrt(252))
                
                # Beta (vs SPY) - simplified calculation
                try:
                    spy_data = await self._get_stock_data('SPY', '1mo', '1d')
                    if spy_data is not None and len(spy_data) == len(data):
                        spy_returns = spy_data['Close'].pct_change().dropna()
                        if len(returns) == len(spy_returns):
                            covariance = np.cov(returns, spy_returns)[0][1]
                            spy_variance = np.var(spy_returns)
                            if spy_variance > 0:
                                risk_metrics['beta'] = covariance / spy_variance
                except:
                    pass  # Beta calculation failed
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _calculate_technical_score(self, indicators: Dict, patterns: Dict, 
                                       trading_signals: Dict) -> Dict[str, Any]:
        """Calculate overall technical analysis score"""
        try:
            scores = {
                'day_trading': 0.0,
                'momentum_trading': 0.0,
                'long_term': 0.0,
                'overall': 0.0
            }
            
            # Extract signal strengths
            for trading_type in ['day_trading', 'momentum_trading', 'long_term']:
                if trading_type in trading_signals:
                    signal = trading_signals[trading_type]
                    strength = signal.get('signal_strength', 0.0)
                    confidence = signal.get('confidence', 0.0)
                    direction = signal.get('direction', 0)
                    
                    # Combine strength, confidence, and direction
                    final_score = strength * confidence * abs(direction) if direction != 0 else 0
                    scores[trading_type] = final_score
            
            # Overall score (weighted average)
            weights = {'day_trading': 0.3, 'momentum_trading': 0.4, 'long_term': 0.3}
            scores['overall'] = sum(scores[t] * weights[t] for t in weights.keys())
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return {'day_trading': 0.0, 'momentum_trading': 0.0, 'long_term': 0.0, 'overall': 0.0}
    
    # ============================================================================
    # COMPREHENSIVE TECHNICAL INDICATORS (From technical signals.txt)
    # ============================================================================
    
    async def _calculate_trend_indicators_comprehensive(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive trend indicators"""
        try:
            trend_data = {}
            
            # Weighted Moving Average
            for period in self.indicators_config.get('wma_periods', [10, 20]):
                if len(data) >= period:
                    weights = np.arange(1, period + 1)
                    wma = data['Close'].rolling(period).apply(
                        lambda prices: np.average(prices, weights=weights), raw=True
                    )
                    trend_data[f'WMA_{period}'] = wma.iloc[-1]
            
            # Hull Moving Average
            hull_period = self.indicators_config.get('hull_ma_period', 20)
            if len(data) >= hull_period:
                half_period = int(hull_period / 2)
                sqrt_period = int(np.sqrt(hull_period))
                wma_half = data['Close'].rolling(half_period).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
                )
                wma_full = data['Close'].rolling(hull_period).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
                )
                hull_raw = (2 * wma_half - wma_full).rolling(sqrt_period).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
                )
                trend_data['HULL_MA'] = hull_raw.iloc[-1]
            
            # Parabolic SAR
            if len(data) >= 20:
                sar_step = self.indicators_config.get('parabolic_sar_step', 0.02)
                sar_max = self.indicators_config.get('parabolic_sar_max', 0.2)
                trend_data['PSAR'] = await self._calculate_parabolic_sar(data, sar_step, sar_max)
            
            # SuperTrend
            st_period = self.indicators_config.get('supertrend_period', 14)
            st_multiplier = self.indicators_config.get('supertrend_multiplier', 3)
            if len(data) >= st_period:
                trend_data.update(await self._calculate_supertrend(data, st_period, st_multiplier))
            
            # TEMA (Triple Exponential Moving Average)
            tema_period = self.indicators_config.get('tema_period', 14)
            if len(data) >= tema_period * 3:
                ema1 = data['Close'].ewm(span=tema_period).mean()
                ema2 = ema1.ewm(span=tema_period).mean()
                ema3 = ema2.ewm(span=tema_period).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                trend_data['TEMA'] = tema.iloc[-1]
            
            # DEMA (Double Exponential Moving Average)
            dema_period = self.indicators_config.get('dema_period', 14)
            if len(data) >= dema_period * 2:
                ema1 = data['Close'].ewm(span=dema_period).mean()
                ema2 = ema1.ewm(span=dema_period).mean()
                dema = 2 * ema1 - ema2
                trend_data['DEMA'] = dema.iloc[-1]
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive trend indicators: {e}")
            return {}
    
    async def _calculate_momentum_indicators_comprehensive(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive momentum indicators"""
        try:
            momentum_data = {}
            
            # Williams %R
            wr_period = self.indicators_config.get('williams_r_period', 14)
            if len(data) >= wr_period:
                highest_high = data['High'].rolling(wr_period).max()
                lowest_low = data['Low'].rolling(wr_period).min()
                williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
                momentum_data['WILLIAMS_R'] = williams_r.iloc[-1]
                
                # Williams %R signals
                if williams_r.iloc[-1] > -20:
                    momentum_data['WILLIAMS_R_signal'] = -1  # Overbought
                elif williams_r.iloc[-1] < -80:
                    momentum_data['WILLIAMS_R_signal'] = 1   # Oversold
                else:
                    momentum_data['WILLIAMS_R_signal'] = 0
            
            # Momentum Indicator
            mom_period = self.indicators_config.get('momentum_period', 12)
            if len(data) >= mom_period:
                momentum = data['Close'] / data['Close'].shift(mom_period) * 100
                momentum_data['MOMENTUM'] = momentum.iloc[-1]
            
            # Rate of Change (ROC)
            roc_period = self.indicators_config.get('roc_period', 12)
            if len(data) >= roc_period:
                roc = ((data['Close'] - data['Close'].shift(roc_period)) / data['Close'].shift(roc_period)) * 100
                momentum_data['ROC'] = roc.iloc[-1]
            
            # TRIX (Triple Exponential Oscillator)
            trix_period = self.indicators_config.get('trix_period', 14)
            if len(data) >= trix_period * 3:
                ema1 = data['Close'].ewm(span=trix_period).mean()
                ema2 = ema1.ewm(span=trix_period).mean()
                ema3 = ema2.ewm(span=trix_period).mean()
                trix = ema3.pct_change() * 10000
                momentum_data['TRIX'] = trix.iloc[-1]
            
            # Ultimate Oscillator
            short_period = self.indicators_config.get('ultimate_osc_short', 7)
            medium_period = self.indicators_config.get('ultimate_osc_medium', 14)
            long_period = self.indicators_config.get('ultimate_osc_long', 28)
            
            if len(data) >= long_period:
                momentum_data['ULTIMATE_OSC'] = await self._calculate_ultimate_oscillator(
                    data, short_period, medium_period, long_period
                )
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive momentum indicators: {e}")
            return {}
    
    async def _calculate_volatility_indicators_comprehensive(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive volatility indicators"""
        try:
            vol_data = {}
            
            # Keltner Channels
            keltner_period = self.indicators_config.get('keltner_period', 20)
            keltner_mult = self.indicators_config.get('keltner_multiplier', 2)
            if len(data) >= keltner_period:
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                keltner_basis = typical_price.rolling(keltner_period).mean()
                atr = await self._calculate_atr_series(data, keltner_period)
                keltner_upper = keltner_basis + (keltner_mult * atr)
                keltner_lower = keltner_basis - (keltner_mult * atr)
                
                vol_data['KELTNER_UPPER'] = keltner_upper.iloc[-1]
                vol_data['KELTNER_BASIS'] = keltner_basis.iloc[-1]
                vol_data['KELTNER_LOWER'] = keltner_lower.iloc[-1]
            
            # Donchian Channels
            donchian_period = self.indicators_config.get('donchian_period', 20)
            if len(data) >= donchian_period:
                donchian_upper = data['High'].rolling(donchian_period).max()
                donchian_lower = data['Low'].rolling(donchian_period).min()
                donchian_basis = (donchian_upper + donchian_lower) / 2
                
                vol_data['DONCHIAN_UPPER'] = donchian_upper.iloc[-1]
                vol_data['DONCHIAN_BASIS'] = donchian_basis.iloc[-1]
                vol_data['DONCHIAN_LOWER'] = donchian_lower.iloc[-1]
            
            # Chaikin Volatility
            chaikin_period = self.indicators_config.get('chaikin_vol_period', 14)
            if len(data) >= chaikin_period:
                hl_diff = data['High'] - data['Low']
                ema_hl = hl_diff.ewm(span=chaikin_period).mean()
                chaikin_vol = ((ema_hl - ema_hl.shift(chaikin_period)) / ema_hl.shift(chaikin_period)) * 100
                vol_data['CHAIKIN_VOLATILITY'] = chaikin_vol.iloc[-1]
            
            return vol_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive volatility indicators: {e}")
            return {}
    
    async def _calculate_volume_indicators_comprehensive(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive volume indicators"""
        try:
            vol_data = {}
            
            # Chaikin Money Flow
            cmf_period = self.indicators_config.get('cmf_period', 20)
            if len(data) >= cmf_period:
                money_flow_mult = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
                money_flow_volume = money_flow_mult * data['Volume']
                cmf = money_flow_volume.rolling(cmf_period).sum() / data['Volume'].rolling(cmf_period).sum()
                vol_data['CMF'] = cmf.iloc[-1]
            
            # Accumulation/Distribution Line
            ad_line = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
            vol_data['AD_LINE'] = ad_line.cumsum().iloc[-1]
            
            # Money Flow Index (MFI)
            mfi_period = self.indicators_config.get('mfi_period', 14)
            if len(data) >= mfi_period:
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                money_flow = typical_price * data['Volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(mfi_period).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(mfi_period).sum()
                
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))
                vol_data['MFI'] = mfi.iloc[-1]
            
            # Ease of Movement (EOM)
            eom_period = self.indicators_config.get('eom_period', 14)
            if len(data) >= eom_period:
                distance_moved = ((data['High'] + data['Low']) / 2) - ((data['High'].shift(1) + data['Low'].shift(1)) / 2)
                box_height = data['Volume'] / (data['High'] - data['Low'])
                eom = distance_moved / box_height
                vol_data['EOM'] = eom.rolling(eom_period).mean().iloc[-1]
            
            # Volume Weighted Average Price (VWAP)
            vwap_period = self.indicators_config.get('vwap_period', 20)
            if len(data) >= vwap_period:
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                vwap = (typical_price * data['Volume']).rolling(vwap_period).sum() / data['Volume'].rolling(vwap_period).sum()
                vol_data['VWAP'] = vwap.iloc[-1]
            
            return vol_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive volume indicators: {e}")
            return {}
    
    async def _calculate_oscillators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate oscillator indicators"""
        try:
            osc_data = {}
            
            # Percentage Price Oscillator (PPO)
            ppo_fast = self.indicators_config.get('ppo_fast', 12)
            ppo_slow = self.indicators_config.get('ppo_slow', 26)
            ppo_signal = self.indicators_config.get('ppo_signal', 9)
            
            if len(data) >= ppo_slow:
                ema_fast = data['Close'].ewm(span=ppo_fast).mean()
                ema_slow = data['Close'].ewm(span=ppo_slow).mean()
                ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
                ppo_signal_line = ppo_line.ewm(span=ppo_signal).mean()
                ppo_histogram = ppo_line - ppo_signal_line
                
                osc_data['PPO'] = ppo_line.iloc[-1]
                osc_data['PPO_SIGNAL'] = ppo_signal_line.iloc[-1]
                osc_data['PPO_HISTOGRAM'] = ppo_histogram.iloc[-1]
            
            # Commodity Channel Index (CCI)
            cci_period = self.indicators_config.get('cci_period', 20)
            if len(data) >= cci_period:
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                sma_tp = typical_price.rolling(cci_period).mean()
                mean_deviation = typical_price.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
                cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
                osc_data['CCI'] = cci.iloc[-1]
            
            # Detrended Price Oscillator
            dpo_period = self.indicators_config.get('dpo_period', 20)
            if len(data) >= dpo_period:
                sma = data['Close'].rolling(dpo_period).mean()
                shift_period = int(dpo_period / 2) + 1
                dpo = data['Close'] - sma.shift(shift_period)
                osc_data['DPO'] = dpo.iloc[-1]
            
            # Fisher Transform
            fisher_period = self.indicators_config.get('fisher_period', 10)
            if len(data) >= fisher_period:
                fisher_data = await self._calculate_fisher_transform(data, fisher_period)
                osc_data.update(fisher_data)
            
            return osc_data
            
        except Exception as e:
            logger.error(f"Error calculating oscillators: {e}")
            return {}
    
    async def _calculate_pattern_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate pattern recognition indicators"""
        try:
            pattern_data = {}
            
            # Pivot Points
            pivot_lookback = self.indicators_config.get('pivot_lookback', 5)
            if len(data) >= pivot_lookback * 2:
                pivots = await self._detect_pivot_points(data, pivot_lookback)
                pattern_data.update(pivots)
            
            # Fibonacci Retracements
            fib_lookback = self.indicators_config.get('fibonacci_lookback', 50)
            if len(data) >= fib_lookback:
                fib_levels = await self._calculate_fibonacci_levels(data, fib_lookback)
                pattern_data.update(fib_levels)
            
            # Candlestick Patterns
            candle_patterns = await self._detect_candlestick_patterns(data)
            pattern_data.update(candle_patterns)
            
            return pattern_data
            
        except Exception as e:
            logger.error(f"Error calculating pattern indicators: {e}")
            return {}
    
    # Helper methods for complex calculations
    async def _calculate_parabolic_sar(self, data: pd.DataFrame, step: float, max_step: float) -> float:
        """Calculate Parabolic SAR"""
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Simplified SAR calculation
            sar = np.zeros(len(close))
            trend = np.ones(len(close))
            af = step
            ep = high[0]
            sar[0] = low[0]
            
            for i in range(1, len(close)):
                if trend[i-1] == 1:  # Uptrend
                    sar[i] = sar[i-1] + af * (ep - sar[i-1])
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + step, max_step)
                    if low[i] <= sar[i]:
                        trend[i] = -1
                        sar[i] = ep
                        af = step
                        ep = low[i]
                    else:
                        trend[i] = 1
                else:  # Downtrend
                    sar[i] = sar[i-1] + af * (ep - sar[i-1])
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + step, max_step)
                    if high[i] >= sar[i]:
                        trend[i] = 1
                        sar[i] = ep
                        af = step
                        ep = high[i]
                    else:
                        trend[i] = -1
            
            return sar[-1]
            
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}")
            return 0.0
    
    async def _calculate_supertrend(self, data: pd.DataFrame, period: int, multiplier: float) -> Dict[str, float]:
        """Calculate SuperTrend indicator"""
        try:
            # Calculate ATR
            atr = await self._calculate_atr_series(data, period)
            
            # Calculate basic upper and lower bands
            hl_avg = (data['High'] + data['Low']) / 2
            upper_band = hl_avg + (multiplier * atr)
            lower_band = hl_avg - (multiplier * atr)
            
            # Calculate final upper and lower bands
            final_upper = upper_band.copy()
            final_lower = lower_band.copy()
            
            for i in range(1, len(data)):
                if upper_band.iloc[i] < final_upper.iloc[i-1] or data['Close'].iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]
                    
                if lower_band.iloc[i] > final_lower.iloc[i-1] or data['Close'].iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]
            
            # Calculate SuperTrend
            supertrend = pd.Series(index=data.index, dtype=float)
            direction = pd.Series(index=data.index, dtype=int)
            
            for i in range(len(data)):
                if i == 0:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    if supertrend.iloc[i-1] == final_upper.iloc[i-1] and data['Close'].iloc[i] <= final_upper.iloc[i]:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        direction.iloc[i] = -1
                    elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and data['Close'].iloc[i] > final_upper.iloc[i]:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        direction.iloc[i] = 1
                    elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and data['Close'].iloc[i] >= final_lower.iloc[i]:
                        supertrend.iloc[i] = final_lower.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = final_upper.iloc[i]
                        direction.iloc[i] = -1
            
            return {
                'SUPERTREND': supertrend.iloc[-1],
                'SUPERTREND_DIRECTION': direction.iloc[-1],
                'SUPERTREND_UPPER': final_upper.iloc[-1],
                'SUPERTREND_LOWER': final_lower.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return {}
    
    async def _calculate_atr_series(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR as a series"""
        try:
            high_low = data['High'] - data['Low']
            high_close_prev = np.abs(data['High'] - data['Close'].shift())
            low_close_prev = np.abs(data['Low'] - data['Close'].shift())
            
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR series: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    async def _calculate_ultimate_oscillator(self, data: pd.DataFrame, short: int, medium: int, long: int) -> float:
        """Calculate Ultimate Oscillator"""
        try:
            buying_pressure = data['Close'] - np.minimum(data['Low'], data['Close'].shift(1))
            true_range = np.maximum(data['High'], data['Close'].shift(1)) - np.minimum(data['Low'], data['Close'].shift(1))
            
            bp_short = buying_pressure.rolling(short).sum()
            tr_short = true_range.rolling(short).sum()
            
            bp_medium = buying_pressure.rolling(medium).sum()
            tr_medium = true_range.rolling(medium).sum()
            
            bp_long = buying_pressure.rolling(long).sum()
            tr_long = true_range.rolling(long).sum()
            
            uo = 100 * ((4 * bp_short / tr_short) + (2 * bp_medium / tr_medium) + (bp_long / tr_long)) / 7
            
            return uo.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating Ultimate Oscillator: {e}")
            return 0.0
    
    async def _calculate_fisher_transform(self, data: pd.DataFrame, period: int) -> Dict[str, float]:
        """Calculate Fisher Transform"""
        try:
            high = data['High'].rolling(period).max()
            low = data['Low'].rolling(period).min()
            
            # Normalize price
            value = 2 * ((data['Close'] - low) / (high - low)) - 1
            value = value.fillna(0).clip(-0.99, 0.99)
            
            # Calculate Fisher Transform
            fisher = pd.Series(index=data.index, dtype=float)
            fisher.iloc[0] = 0
            
            for i in range(1, len(value)):
                fisher.iloc[i] = 0.5 * np.log((1 + value.iloc[i]) / (1 - value.iloc[i])) + 0.5 * fisher.iloc[i-1]
            
            return {
                'FISHER_TRANSFORM': fisher.iloc[-1],
                'FISHER_SIGNAL': fisher.shift(1).iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fisher Transform: {e}")
            return {}
    
    async def _detect_pivot_points(self, data: pd.DataFrame, lookback: int) -> Dict[str, Any]:
        """Detect pivot points"""
        try:
            pivot_data = {}
            
            # Calculate traditional pivot points
            if len(data) >= 1:
                high = data['High'].iloc[-1]
                low = data['Low'].iloc[-1]
                close = data['Close'].iloc[-1]
                
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                s1 = 2 * pivot - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                
                pivot_data.update({
                    'PIVOT_POINT': pivot,
                    'RESISTANCE_1': r1,
                    'SUPPORT_1': s1,
                    'RESISTANCE_2': r2,
                    'SUPPORT_2': s2
                })
            
            return pivot_data
            
        except Exception as e:
            logger.error(f"Error detecting pivot points: {e}")
            return {}
    
    async def _calculate_fibonacci_levels(self, data: pd.DataFrame, lookback: int) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            fib_data = {}
            
            if len(data) >= lookback:
                recent_data = data.iloc[-lookback:]
                high = recent_data['High'].max()
                low = recent_data['Low'].min()
                
                diff = high - low
                
                fib_data.update({
                    'FIB_0': high,
                    'FIB_23_6': high - (0.236 * diff),
                    'FIB_38_2': high - (0.382 * diff),
                    'FIB_50': high - (0.5 * diff),
                    'FIB_61_8': high - (0.618 * diff),
                    'FIB_100': low
                })
            
            return fib_data
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    async def _detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Detect basic candlestick patterns"""
        try:
            patterns = {}
            
            if len(data) >= 3:
                current = data.iloc[-1]
                previous = data.iloc[-2]
                
                open_price = current['Open']
                close = current['Close']
                high = current['High']
                low = current['Low']
                
                # Doji pattern
                body_size = abs(close - open_price)
                total_range = high - low
                patterns['DOJI'] = body_size <= (total_range * 0.1) if total_range > 0 else False
                
                # Hammer pattern
                lower_shadow = min(open_price, close) - low
                upper_shadow = high - max(open_price, close)
                patterns['HAMMER'] = (lower_shadow > body_size * 2) and (upper_shadow < body_size * 0.5)
                
                # Engulfing patterns
                if len(data) >= 2:
                    prev_open = previous['Open']
                    prev_close = previous['Close']
                    
                    # Bullish engulfing
                    patterns['BULLISH_ENGULFING'] = (prev_close < prev_open and 
                                                   close > open_price and 
                                                   close > prev_open and 
                                                   open_price < prev_close)
                    
                    # Bearish engulfing
                    patterns['BEARISH_ENGULFING'] = (prev_close > prev_open and 
                                                   close < open_price and 
                                                   close < prev_open and 
                                                   open_price > prev_close)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {}

    async def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
        logger.info("Technical Analyzer cleaned up")
