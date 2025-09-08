"""
Automated Trading Signal Generation
Generates actionable buy/sell signals with confidence scores
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yfinance as yf

from shared.models import Catalyst, CatalystType, SentimentLabel
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.fundamental_analysis import FundamentalAnalyzer
from analysis.predictive_impact_scorer import PredictiveImpactScorer, PredictiveScore

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TradingSignal:
    """Comprehensive trading signal with all relevant information"""
    ticker: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: str  # 'intraday', 'swing', 'position'
    
    # Signal sources
    catalyst_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    
    # Supporting information
    reasoning: List[str]
    key_catalysts: List[Catalyst]
    technical_indicators: Dict[str, Any]
    risk_factors: List[str]
    
    # Metadata
    generated_at: datetime
    expires_at: datetime
    signal_id: str

@dataclass
class MarketConditions:
    """Current market conditions for signal generation"""
    overall_trend: str  # 'bullish', 'bearish', 'neutral'
    volatility_level: str  # 'low', 'medium', 'high'
    volume_profile: str  # 'normal', 'high', 'low'
    sector_rotation: Dict[str, float]
    fear_greed_index: float
    market_regime: str  # 'trending', 'ranging', 'volatile'

class AutomatedSignalGenerator:
    """
    Advanced automated trading signal generation system
    Combines catalyst analysis, technical indicators, and fundamental analysis
    """
    
    def __init__(self, config, db_manager, cache_manager):
        self.config = config
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        
        # Initialize analysis components
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.fundamental_analyzer = FundamentalAnalyzer(config)
        self.predictive_scorer = PredictiveImpactScorer(config, db_manager)
        
        # Signal generation parameters
        self.signal_thresholds = {
            'buy': {
                'minimum_confidence': 0.65,
                'minimum_catalyst_impact': 60,
                'technical_bullish_threshold': 0.6,
                'fundamental_positive_threshold': 0.55
            },
            'strong_buy': {
                'minimum_confidence': 0.80,
                'minimum_catalyst_impact': 80,
                'technical_bullish_threshold': 0.75,
                'fundamental_positive_threshold': 0.7
            },
            'sell': {
                'minimum_confidence': 0.60,
                'minimum_catalyst_impact': 60,
                'technical_bearish_threshold': 0.6,
                'fundamental_negative_threshold': 0.55
            }
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_position_risk': 0.02,  # 2% of portfolio
            'min_risk_reward': 2.0,     # Minimum 2:1 risk/reward
            'max_correlation': 0.7,     # Maximum correlation between positions
            'volatility_adjustment': True
        }
        
        # Signal tracking
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history = []
        
        logger.info("Automated Signal Generator initialized")
    
    async def generate_signal(self, catalyst: Catalyst, market_data: Dict = None) -> Optional[TradingSignal]:
        """
        Generate a comprehensive trading signal from a catalyst
        """
        try:
            ticker = catalyst.ticker
            
            # Get current market data
            current_data = await self._get_market_data(ticker, market_data)
            if not current_data:
                logger.warning(f"Could not get market data for {ticker}")
                return None
            
            # Get market conditions
            market_conditions = await self._assess_market_conditions()
            
            # Analyze catalyst impact
            predictive_score = await self.predictive_scorer.predict_impact(catalyst)
            
            # Perform technical analysis
            technical_analysis = await self._perform_technical_analysis(ticker, current_data)
            
            # Perform fundamental analysis
            fundamental_analysis = await self._perform_fundamental_analysis(ticker)
            
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(
                catalyst, predictive_score, technical_analysis, fundamental_analysis
            )
            
            # Generate signal based on all factors
            signal = self._generate_signal_from_analysis(
                ticker, catalyst, predictive_score, technical_analysis, 
                fundamental_analysis, composite_scores, current_data, market_conditions
            )
            
            if signal:
                # Validate and refine signal
                signal = self._validate_and_refine_signal(signal, market_conditions)
                
                # Store signal
                self.active_signals[signal.signal_id] = signal
                self.signal_history.append(signal)
                
                logger.info(f"Generated {signal.signal_type.value} signal for {ticker} with {signal.confidence:.2f} confidence")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return None
    
    async def _get_market_data(self, ticker: str, provided_data: Dict = None) -> Optional[Dict]:
        """Get comprehensive market data for the ticker"""
        try:
            if provided_data:
                return provided_data
            
            # Check cache first
            cache_key = f"market_data_{ticker}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            
            # Get current price data
            hist = stock.history(period="5d")
            if hist.empty:
                return None
            
            info = stock.info
            current_price = hist['Close'].iloc[-1]
            
            market_data = {
                'current_price': current_price,
                'volume': hist['Volume'].iloc[-1],
                'avg_volume': hist['Volume'].mean(),
                'high_52w': info.get('fiftyTwoWeekHigh', current_price),
                'low_52w': info.get('fiftyTwoWeekLow', current_price),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'price_history': hist
            }
            
            # Cache for 5 minutes
            self.cache_manager.set(cache_key, market_data, ttl=300)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return None
    
    async def _assess_market_conditions(self) -> MarketConditions:
        """Assess overall market conditions"""
        try:
            # This is a simplified implementation
            # In practice, you'd analyze broader market indices, VIX, etc.
            
            # Get SPY data as market proxy
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1mo")
            
            if spy_hist.empty:
                return self._default_market_conditions()
            
            # Calculate trend
            recent_close = spy_hist['Close'].iloc[-1]
            ma_20 = spy_hist['Close'].rolling(20).mean().iloc[-1]
            
            if recent_close > ma_20 * 1.02:
                overall_trend = 'bullish'
            elif recent_close < ma_20 * 0.98:
                overall_trend = 'bearish'
            else:
                overall_trend = 'neutral'
            
            # Calculate volatility
            returns = spy_hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            if volatility < 0.15:
                volatility_level = 'low'
            elif volatility < 0.25:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            # Volume analysis
            avg_volume = spy_hist['Volume'].mean()
            recent_volume = spy_hist['Volume'].iloc[-1]
            volume_ratio = recent_volume / avg_volume
            
            if volume_ratio > 1.2:
                volume_profile = 'high'
            elif volume_ratio < 0.8:
                volume_profile = 'low'
            else:
                volume_profile = 'normal'
            
            return MarketConditions(
                overall_trend=overall_trend,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                sector_rotation={},  # Simplified
                fear_greed_index=50.0,  # Neutral default
                market_regime='trending' if overall_trend != 'neutral' else 'ranging'
            )
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return self._default_market_conditions()
    
    def _default_market_conditions(self) -> MarketConditions:
        """Default market conditions when analysis fails"""
        return MarketConditions(
            overall_trend='neutral',
            volatility_level='medium',
            volume_profile='normal',
            sector_rotation={},
            fear_greed_index=50.0,
            market_regime='ranging'
        )
    
    async def _perform_technical_analysis(self, ticker: str, market_data: Dict) -> Dict:
        """Perform comprehensive technical analysis"""
        try:
            price_history = market_data.get('price_history')
            if price_history is None or price_history.empty:
                return {'score': 0.5, 'signals': [], 'indicators': {}}
            
            # Use existing technical analyzer
            analysis = self.technical_analyzer.analyze_stock(ticker, price_history)
            
            # Convert to standardized format
            bullish_signals = 0
            bearish_signals = 0
            
            for signal in analysis.get('signals', []):
                if 'bullish' in signal.lower() or 'buy' in signal.lower():
                    bullish_signals += 1
                elif 'bearish' in signal.lower() or 'sell' in signal.lower():
                    bearish_signals += 1
            
            total_signals = bullish_signals + bearish_signals
            if total_signals > 0:
                technical_score = bullish_signals / total_signals
            else:
                technical_score = 0.5  # Neutral
            
            return {
                'score': technical_score,
                'signals': analysis.get('signals', []),
                'indicators': analysis.get('indicators', {}),
                'bullish_count': bullish_signals,
                'bearish_count': bearish_signals
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {'score': 0.5, 'signals': [], 'indicators': {}}
    
    async def _perform_fundamental_analysis(self, ticker: str) -> Dict:
        """Perform fundamental analysis"""
        try:
            # Use existing fundamental analyzer
            analysis = self.fundamental_analyzer.analyze_stock(ticker)
            
            # Extract fundamental score
            fundamental_score = analysis.get('overall_score', 0.5)
            
            return {
                'score': fundamental_score,
                'metrics': analysis.get('metrics', {}),
                'strengths': analysis.get('strengths', []),
                'weaknesses': analysis.get('weaknesses', [])
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {'score': 0.5, 'metrics': {}, 'strengths': [], 'weaknesses': []}
    
    def _calculate_composite_scores(self, catalyst: Catalyst, predictive_score: PredictiveScore,
                                  technical_analysis: Dict, fundamental_analysis: Dict) -> Dict:
        """Calculate composite scores from all analysis components"""
        
        # Catalyst score (0-1)
        catalyst_score = (catalyst.impact / 100.0) * catalyst.confidence
        
        # Adjust catalyst score based on sentiment
        if catalyst.sentiment_label == SentimentLabel.POSITIVE:
            catalyst_sentiment_multiplier = 1.0
        elif catalyst.sentiment_label == SentimentLabel.NEGATIVE:
            catalyst_sentiment_multiplier = -1.0
        else:
            catalyst_sentiment_multiplier = 0.0
        
        catalyst_score *= catalyst_sentiment_multiplier
        
        # Technical score
        technical_score = technical_analysis.get('score', 0.5)
        # Convert to -1 to 1 scale
        technical_score = (technical_score - 0.5) * 2
        
        # Fundamental score
        fundamental_score = fundamental_analysis.get('score', 0.5)
        # Convert to -1 to 1 scale  
        fundamental_score = (fundamental_score - 0.5) * 2
        
        # Predictive score
        predictive_change = predictive_score.predicted_price_change
        predictive_confidence = predictive_score.confidence
        
        # Weighted composite score
        weights = {
            'catalyst': 0.35,
            'predictive': 0.25,
            'technical': 0.25,
            'fundamental': 0.15
        }
        
        composite_score = (
            catalyst_score * weights['catalyst'] +
            (predictive_change * predictive_confidence) * weights['predictive'] +
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental']
        )
        
        return {
            'composite': composite_score,
            'catalyst': catalyst_score,
            'technical': technical_score,
            'fundamental': fundamental_score,
            'predictive': predictive_change,
            'sentiment': catalyst_sentiment_multiplier
        }
    
    def _generate_signal_from_analysis(self, ticker: str, catalyst: Catalyst, 
                                     predictive_score: PredictiveScore,
                                     technical_analysis: Dict, fundamental_analysis: Dict,
                                     composite_scores: Dict, market_data: Dict,
                                     market_conditions: MarketConditions) -> Optional[TradingSignal]:
        """Generate trading signal from comprehensive analysis"""
        
        composite_score = composite_scores['composite']
        current_price = market_data['current_price']
        
        # Determine signal type based on composite score
        if composite_score > 0.15:
            if composite_score > 0.30:
                signal_type = SignalType.STRONG_BUY
                strength = SignalStrength.VERY_STRONG if composite_score > 0.45 else SignalStrength.STRONG
            else:
                signal_type = SignalType.BUY
                strength = SignalStrength.STRONG if composite_score > 0.25 else SignalStrength.MODERATE
        elif composite_score < -0.15:
            if composite_score < -0.30:
                signal_type = SignalType.STRONG_SELL
                strength = SignalStrength.VERY_STRONG if composite_score < -0.45 else SignalStrength.STRONG
            else:
                signal_type = SignalType.SELL
                strength = SignalStrength.STRONG if composite_score < -0.25 else SignalStrength.MODERATE
        else:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK
        
        # Check if signal meets minimum thresholds
        thresholds = self.signal_thresholds.get(signal_type.value, {})
        min_confidence = thresholds.get('minimum_confidence', 0.5)
        
        overall_confidence = self._calculate_signal_confidence(
            catalyst, predictive_score, technical_analysis, fundamental_analysis, composite_scores
        )
        
        if overall_confidence < min_confidence:
            return None
        
        # Calculate entry, target, and stop loss prices
        entry_price = current_price
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # For buy signals
            expected_move = abs(predictive_score.predicted_price_change)
            target_price = entry_price * (1 + expected_move * 1.5)  # 1.5x expected move
            stop_loss = entry_price * (1 - expected_move * 0.5)     # 0.5x expected move
        else:
            # For sell signals
            expected_move = abs(predictive_score.predicted_price_change)
            target_price = entry_price * (1 - expected_move * 1.5)  # 1.5x expected move
            stop_loss = entry_price * (1 + expected_move * 0.5)     # 0.5x expected move
        
        # Calculate risk/reward ratio
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Skip signals with poor risk/reward
        if risk_reward_ratio < self.risk_params['min_risk_reward']:
            return None
        
        # Determine timeframe
        timeframe = self._determine_timeframe(catalyst, market_conditions)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            catalyst, predictive_score, technical_analysis, fundamental_analysis, composite_scores
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            catalyst, market_data, market_conditions, technical_analysis
        )
        
        # Calculate expiration
        timeframe_hours = {'intraday': 1, 'swing': 72, 'position': 168}  # hours
        expires_at = datetime.now() + timedelta(hours=timeframe_hours.get(timeframe, 24))
        
        signal = TradingSignal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=overall_confidence,
            strength=strength,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            timeframe=timeframe,
            
            catalyst_score=composite_scores['catalyst'],
            technical_score=composite_scores['technical'],
            fundamental_score=composite_scores['fundamental'],
            sentiment_score=composite_scores['sentiment'],
            
            reasoning=reasoning,
            key_catalysts=[catalyst],
            technical_indicators=technical_analysis.get('indicators', {}),
            risk_factors=risk_factors,
            
            generated_at=datetime.now(),
            expires_at=expires_at,
            signal_id=f"{ticker}_{catalyst.id}_{int(datetime.now().timestamp())}"
        )
        
        return signal
    
    def _calculate_signal_confidence(self, catalyst: Catalyst, predictive_score: PredictiveScore,
                                   technical_analysis: Dict, fundamental_analysis: Dict,
                                   composite_scores: Dict) -> float:
        """Calculate overall signal confidence"""
        
        # Base confidence from catalyst
        base_confidence = catalyst.confidence
        
        # Boost from predictive model
        predictive_boost = predictive_score.confidence * 0.2
        
        # Boost from technical alignment
        technical_score = technical_analysis.get('score', 0.5)
        technical_alignment = abs(technical_score - 0.5) * 2  # 0 to 1
        technical_boost = technical_alignment * 0.15
        
        # Boost from fundamental alignment
        fundamental_score = fundamental_analysis.get('score', 0.5)
        fundamental_alignment = abs(fundamental_score - 0.5) * 2  # 0 to 1
        fundamental_boost = fundamental_alignment * 0.1
        
        # Composite score strength
        composite_strength = abs(composite_scores['composite'])
        composite_boost = min(composite_strength, 0.15)
        
        total_confidence = base_confidence + predictive_boost + technical_boost + fundamental_boost + composite_boost
        
        return min(total_confidence, 0.95)
    
    def _determine_timeframe(self, catalyst: Catalyst, market_conditions: MarketConditions) -> str:
        """Determine appropriate timeframe for the signal"""
        
        # Base timeframe by catalyst type
        timeframe_map = {
            CatalystType.EARNINGS: 'intraday',
            CatalystType.FDA_APPROVAL: 'swing',
            CatalystType.MERGER_ACQUISITION: 'intraday',
            CatalystType.PRODUCT_LAUNCH: 'swing',
            CatalystType.REGULATORY: 'swing',
            CatalystType.PARTNERSHIP: 'swing',
            CatalystType.INSIDER_TRADING: 'position',
            CatalystType.ANALYST_UPGRADE: 'intraday',
            CatalystType.NEWS: 'intraday'
        }
        
        base_timeframe = timeframe_map.get(catalyst.category, 'swing')
        
        # Adjust based on market conditions
        if market_conditions.volatility_level == 'high':
            # Shorten timeframe in high volatility
            if base_timeframe == 'position':
                return 'swing'
            elif base_timeframe == 'swing':
                return 'intraday'
        
        return base_timeframe
    
    def _generate_reasoning(self, catalyst: Catalyst, predictive_score: PredictiveScore,
                          technical_analysis: Dict, fundamental_analysis: Dict,
                          composite_scores: Dict) -> List[str]:
        """Generate human-readable reasoning for the signal"""
        
        reasoning = []
        
        # Catalyst reasoning
        if catalyst.impact > 70:
            reasoning.append(f"High-impact {catalyst.category.value} catalyst with {catalyst.impact}% impact score")
        
        if catalyst.confidence > 0.8:
            reasoning.append(f"High confidence catalyst detection ({catalyst.confidence:.1%})")
        
        # Predictive reasoning
        if abs(predictive_score.predicted_price_change) > 0.05:
            direction = "upward" if predictive_score.predicted_price_change > 0 else "downward"
            reasoning.append(f"Predictive model indicates {direction} price movement of {abs(predictive_score.predicted_price_change):.1%}")
        
        # Technical reasoning
        technical_signals = technical_analysis.get('signals', [])
        if technical_signals:
            reasoning.append(f"Technical analysis shows: {', '.join(technical_signals[:2])}")
        
        # Fundamental reasoning
        fundamental_strengths = fundamental_analysis.get('strengths', [])
        if fundamental_strengths:
            reasoning.append(f"Fundamental strengths: {', '.join(fundamental_strengths[:2])}")
        
        # Sentiment reasoning
        if catalyst.sentiment_label == SentimentLabel.POSITIVE:
            reasoning.append("Positive market sentiment around catalyst")
        elif catalyst.sentiment_label == SentimentLabel.NEGATIVE:
            reasoning.append("Negative market sentiment around catalyst")
        
        return reasoning
    
    def _identify_risk_factors(self, catalyst: Catalyst, market_data: Dict,
                             market_conditions: MarketConditions, technical_analysis: Dict) -> List[str]:
        """Identify potential risk factors"""
        
        risks = []
        
        # Market condition risks
        if market_conditions.volatility_level == 'high':
            risks.append("High market volatility increases position risk")
        
        if market_conditions.overall_trend == 'bearish' and catalyst.sentiment_label == SentimentLabel.POSITIVE:
            risks.append("Positive catalyst against bearish market trend")
        
        # Technical risks
        bearish_signals = technical_analysis.get('bearish_count', 0)
        if bearish_signals > 2:
            risks.append("Multiple bearish technical indicators present")
        
        # Stock-specific risks
        beta = market_data.get('beta', 1.0)
        if beta > 1.5:
            risks.append(f"High beta stock ({beta:.1f}) amplifies market movements")
        
        # Volume risks
        volume_ratio = market_data.get('volume') / market_data.get('avg_volume', 1)
        if volume_ratio < 0.5:
            risks.append("Below-average trading volume may limit liquidity")
        
        # Catalyst-specific risks
        if catalyst.confidence < 0.7:
            risks.append("Lower confidence in catalyst detection")
        
        return risks
    
    def _validate_and_refine_signal(self, signal: TradingSignal, 
                                   market_conditions: MarketConditions) -> TradingSignal:
        """Validate and refine the generated signal"""
        
        # Adjust confidence based on market conditions
        if market_conditions.volatility_level == 'high':
            signal.confidence *= 0.9  # Reduce confidence in high volatility
        
        if market_conditions.volume_profile == 'low':
            signal.confidence *= 0.95  # Slightly reduce confidence with low volume
        
        # Adjust target and stop loss based on volatility
        if market_conditions.volatility_level == 'high':
            # Widen stops in high volatility
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                signal.stop_loss *= 0.98  # Move stop loss further away
            else:
                signal.stop_loss *= 1.02
        
        # Recalculate risk/reward ratio
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            risk = signal.entry_price - signal.stop_loss
            reward = signal.target_price - signal.entry_price
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.target_price
        
        signal.risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return signal
    
    def get_active_signals(self, ticker: str = None) -> List[TradingSignal]:
        """Get currently active signals"""
        active = []
        current_time = datetime.now()
        
        for signal_id, signal in self.active_signals.items():
            if signal.expires_at > current_time:
                if ticker is None or signal.ticker == ticker:
                    active.append(signal)
            else:
                # Remove expired signals
                del self.active_signals[signal_id]
        
        return active
    
    def update_signal_outcome(self, signal_id: str, outcome: Dict):
        """Update signal with actual outcome for performance tracking"""
        try:
            # Store outcome for performance analysis
            outcome_data = {
                'signal_id': signal_id,
                'outcome': outcome,
                'timestamp': datetime.now()
            }
            
            # TODO: Save to database for analysis  
            # self.db_manager.store_signal_outcome(outcome_data)
            logger.info("Signal outcome recorded (database storage to be implemented)")
            
            logger.info(f"Stored outcome for signal {signal_id}")
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get signal generation performance statistics"""
        try:
            total_signals = len(self.signal_history)
            
            if total_signals == 0:
                return {'total_signals': 0}
            
            # Calculate basic stats
            signal_types = {}
            confidence_sum = 0
            
            for signal in self.signal_history:
                signal_type = signal.signal_type.value
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
                confidence_sum += signal.confidence
            
            avg_confidence = confidence_sum / total_signals
            
            return {
                'total_signals': total_signals,
                'active_signals': len(self.active_signals),
                'signal_distribution': signal_types,
                'average_confidence': avg_confidence,
                'last_signal_time': self.signal_history[-1].generated_at if self.signal_history else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {'error': str(e)}