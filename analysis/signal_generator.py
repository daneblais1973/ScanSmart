import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

from shared.models import Catalyst, CatalystType, SourceType
from core.config import AppConfig
from core.database import DatabaseManager
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.fundamental_analysis import FundamentalAnalyzer
from nlp.advanced_models import AdvancedNLPProcessor

logger = logging.getLogger(__name__)

class AITradingSignalGenerator:
    """AI-powered trading signal generator combining technical, fundamental, and sentiment analysis"""
    
    def __init__(self, config: AppConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        
        # Initialize analysis modules
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.fundamental_analyzer = FundamentalAnalyzer(config)
        self.nlp_processor = AdvancedNLPProcessor(config)
        
        self._executor = ThreadPoolExecutor(max_workers=6)
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Signal generation weights for different trading styles
        self.signal_weights = {
            'day_trading': {
                'technical': 0.60,
                'sentiment': 0.25,
                'catalyst': 0.15,
                'fundamental': 0.05  # Less important for day trading
            },
            'momentum_trading': {
                'technical': 0.45,
                'sentiment': 0.25,
                'catalyst': 0.20,
                'fundamental': 0.10
            },
            'long_term': {
                'fundamental': 0.40,
                'technical': 0.25,
                'catalyst': 0.20,
                'sentiment': 0.15
            }
        }
        
        # Risk adjustment factors
        self.risk_factors = {
            'market_cap': {
                'mega_cap': 0.8,     # > $200B - Lower risk
                'large_cap': 0.9,    # $10B - $200B
                'mid_cap': 1.0,      # $2B - $10B - Baseline
                'small_cap': 1.2,    # $300M - $2B - Higher risk
                'micro_cap': 1.5     # < $300M - Much higher risk
            },
            'volatility': {
                'low': 0.8,      # < 20% annualized
                'medium': 1.0,   # 20% - 40%
                'high': 1.3,     # 40% - 60%
                'extreme': 1.6   # > 60%
            },
            'sector': {
                'Utilities': 0.8,
                'Consumer Defensive': 0.9,
                'Healthcare': 0.9,
                'Financial Services': 1.0,
                'Technology': 1.1,
                'Consumer Cyclical': 1.1,
                'Energy': 1.2,
                'Biotechnology': 1.4
            }
        }
        
        logger.info("AI Trading Signal Generator initialized")
    
    async def generate_comprehensive_signals(self, ticker: str, 
                                           catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate comprehensive trading signals for a stock"""
        try:
            logger.info(f"Generating comprehensive signals for {ticker}")
            
            # Run all analyses in parallel
            analyses = await asyncio.gather(
                self._run_technical_analysis(ticker, catalyst),
                self._run_fundamental_analysis(ticker, catalyst),
                self._run_sentiment_analysis(ticker, catalyst),
                self._calculate_catalyst_impact(ticker, catalyst),
                return_exceptions=True
            )
            
            technical_analysis, fundamental_analysis, sentiment_analysis, catalyst_impact = analyses
            
            # Handle any analysis failures
            if isinstance(technical_analysis, Exception):
                logger.error(f"Technical analysis failed for {ticker}: {technical_analysis}")
                technical_analysis = {}
            
            if isinstance(fundamental_analysis, Exception):
                logger.error(f"Fundamental analysis failed for {ticker}: {fundamental_analysis}")
                fundamental_analysis = {}
            
            if isinstance(sentiment_analysis, Exception):
                logger.error(f"Sentiment analysis failed for {ticker}: {sentiment_analysis}")
                sentiment_analysis = {}
            
            if isinstance(catalyst_impact, Exception):
                logger.error(f"Catalyst impact calculation failed for {ticker}: {catalyst_impact}")
                catalyst_impact = {}
            
            # Generate AI-powered signals
            ai_signals = await self._generate_ai_signals(
                ticker, technical_analysis, fundamental_analysis, 
                sentiment_analysis, catalyst_impact, catalyst
            )
            
            # Calculate risk-adjusted scores
            risk_adjusted_signals = await self._apply_risk_adjustments(
                ai_signals, technical_analysis, fundamental_analysis
            )
            
            # Store analyses in database
            await self._store_analyses(ticker, technical_analysis, fundamental_analysis)
            
            # Generate final trading recommendations
            final_signals = await self._generate_final_recommendations(
                ticker, risk_adjusted_signals, technical_analysis, 
                fundamental_analysis, catalyst
            )
            
            return {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'fundamental_analysis': fundamental_analysis,
                'sentiment_analysis': sentiment_analysis,
                'catalyst_impact': catalyst_impact,
                'ai_signals': risk_adjusted_signals,
                'final_recommendations': final_signals,
                'data_quality_score': self._calculate_data_quality_score(
                    technical_analysis, fundamental_analysis, sentiment_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals for {ticker}: {e}")
            return {'error': str(e), 'ticker': ticker}
    
    async def _run_technical_analysis(self, ticker: str, 
                                    catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Run technical analysis"""
        try:
            # Use different timeframes based on catalyst urgency
            if catalyst and catalyst.impact >= 80:
                period = '5d'    # Short-term for high-impact catalysts
                interval = '15m'
            elif catalyst and catalyst.impact >= 60:
                period = '1mo'
                interval = '1h'
            else:
                period = '3mo'   # Standard analysis
                interval = '1d'
            
            analysis = await self.technical_analyzer.analyze_stock_technical(
                ticker, period=period, interval=interval, catalyst=catalyst
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {ticker}: {e}")
            return {}
    
    async def _run_fundamental_analysis(self, ticker: str, 
                                      catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Run fundamental analysis"""
        try:
            analysis = await self.fundamental_analyzer.analyze_stock_fundamentals(
                ticker, catalyst=catalyst
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {ticker}: {e}")
            return {}
    
    async def _run_sentiment_analysis(self, ticker: str, 
                                     catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Run sentiment analysis on recent news and catalysts"""
        try:
            # Get recent catalysts for this ticker
            recent_catalysts = self.db_manager.get_catalysts(
                ticker=ticker, 
                limit=20
            )
            
            if not recent_catalysts:
                return {'overall_sentiment': 0.0, 'confidence': 0.0, 'catalyst_count': 0}
            
            # Analyze sentiment using advanced NLP
            sentiment_scores = []
            confidence_scores = []
            
            for cat in recent_catalysts:
                if cat.catalyst:  # Ensure catalyst text exists
                    try:
                        # Use advanced NLP models for sentiment analysis
                        analysis = await self.nlp_processor.analyze_financial_text(
                            cat.catalyst, ticker
                        )
                        
                        sentiment_scores.append(analysis.get('sentiment_score', 0.0))
                        confidence_scores.append(analysis.get('confidence', 0.0))
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze catalyst sentiment: {e}")
                        continue
            
            # Calculate weighted sentiment (more recent catalysts have higher weight)
            if sentiment_scores:
                weights = np.exp(np.linspace(-1, 0, len(sentiment_scores)))  # Exponential decay
                weighted_sentiment = np.average(sentiment_scores, weights=weights)
                avg_confidence = np.mean(confidence_scores)
            else:
                weighted_sentiment = 0.0
                avg_confidence = 0.0
            
            # Include current catalyst if provided
            if catalyst and catalyst.sentiment_score is not None:
                # Give current catalyst higher weight
                if sentiment_scores:
                    weighted_sentiment = weighted_sentiment * 0.7 + catalyst.sentiment_score * 0.3
                    avg_confidence = avg_confidence * 0.7 + catalyst.confidence * 0.3
                else:
                    weighted_sentiment = catalyst.sentiment_score
                    avg_confidence = catalyst.confidence
            
            return {
                'overall_sentiment': weighted_sentiment,
                'confidence': avg_confidence,
                'catalyst_count': len(recent_catalysts),
                'recent_sentiment_trend': self._calculate_sentiment_trend(recent_catalysts)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {ticker}: {e}")
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'catalyst_count': 0}
    
    async def _calculate_catalyst_impact(self, ticker: str, 
                                       catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Calculate expected catalyst impact using historical data"""
        try:
            if not catalyst:
                return {'impact_score': 0.0, 'confidence': 0.0, 'time_horizon': 'unknown'}
            
            # Get historical correlation data for similar catalysts
            correlation_data = self.db_manager.get_catalyst_correlation_history(
                ticker=ticker,
                category=catalyst.category.value if catalyst.category else None
            )
            
            if not correlation_data:
                # Use catalyst's stated impact as baseline
                return {
                    'impact_score': catalyst.impact / 100.0,
                    'confidence': catalyst.confidence,
                    'time_horizon': self._estimate_time_horizon(catalyst),
                    'historical_data': False
                }
            
            # Calculate weighted impact based on historical performance
            historical_impacts = [data['historical_impact'] for data in correlation_data]
            success_rates = [data['success_rate'] for data in correlation_data]
            
            if historical_impacts and success_rates:
                avg_historical_impact = np.mean(historical_impacts)
                avg_success_rate = np.mean(success_rates)
                
                # Adjust catalyst impact based on historical performance
                adjusted_impact = (
                    catalyst.impact / 100.0 * 0.6 +  # Current catalyst weight
                    avg_historical_impact * 0.4       # Historical average weight
                ) * avg_success_rate  # Adjust by success rate
                
                confidence = min(1.0, catalyst.confidence + (avg_success_rate - 0.5) * 0.2)
            else:
                adjusted_impact = catalyst.impact / 100.0
                confidence = catalyst.confidence
            
            return {
                'impact_score': adjusted_impact,
                'confidence': confidence,
                'time_horizon': self._estimate_time_horizon(catalyst),
                'historical_data': len(correlation_data) > 0,
                'similar_catalysts_count': len(correlation_data)
            }
            
        except Exception as e:
            logger.error(f"Catalyst impact calculation failed for {ticker}: {e}")
            return {'impact_score': 0.0, 'confidence': 0.0, 'time_horizon': 'unknown'}
    
    async def _generate_ai_signals(self, ticker: str, technical_analysis: Dict,
                                 fundamental_analysis: Dict, sentiment_analysis: Dict,
                                 catalyst_impact: Dict, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate AI-powered trading signals combining all analyses"""
        try:
            ai_signals = {}
            
            # Generate signals for each trading type
            for trading_type in ['day_trading', 'momentum_trading', 'long_term']:
                signal = await self._generate_trading_type_signal(
                    trading_type, ticker, technical_analysis, fundamental_analysis,
                    sentiment_analysis, catalyst_impact, catalyst
                )
                ai_signals[trading_type] = signal
            
            # Generate overall recommendation
            overall_signal = await self._generate_overall_signal(
                ai_signals, technical_analysis, fundamental_analysis
            )
            ai_signals['overall'] = overall_signal
            
            return ai_signals
            
        except Exception as e:
            logger.error(f"AI signal generation failed for {ticker}: {e}")
            return {}
    
    async def _generate_trading_type_signal(self, trading_type: str, ticker: str,
                                          technical_analysis: Dict, fundamental_analysis: Dict,
                                          sentiment_analysis: Dict, catalyst_impact: Dict,
                                          catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate signal for specific trading type"""
        try:
            weights = self.signal_weights.get(trading_type, self.signal_weights['momentum_trading'])
            
            # Extract component scores
            technical_score = self._extract_technical_score(technical_analysis, trading_type)
            fundamental_score = self._extract_fundamental_score(fundamental_analysis, trading_type)
            sentiment_score = sentiment_analysis.get('overall_sentiment', 0.0)
            catalyst_score = catalyst_impact.get('impact_score', 0.0)
            
            # Apply weights to generate combined score
            combined_score = (
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                sentiment_score * weights['sentiment'] +
                catalyst_score * weights['catalyst']
            )
            
            # Calculate confidence based on data quality and agreement
            confidence = self._calculate_signal_confidence(
                technical_analysis, fundamental_analysis, sentiment_analysis, catalyst_impact
            )
            
            # Determine signal direction and strength
            if combined_score > 0.15:
                direction = 1  # Bullish
                signal_strength = min(1.0, combined_score * 2)
            elif combined_score < -0.15:
                direction = -1  # Bearish
                signal_strength = min(1.0, abs(combined_score) * 2)
            else:
                direction = 0  # Neutral
                signal_strength = 0.0
            
            # Generate entry/exit prices based on technical analysis
            entry_exit_prices = self._calculate_entry_exit_prices(
                technical_analysis, direction, trading_type
            )
            
            return {
                'signal_type': trading_type,
                'direction': direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'combined_score': combined_score,
                'component_scores': {
                    'technical': technical_score,
                    'fundamental': fundamental_score,
                    'sentiment': sentiment_score,
                    'catalyst': catalyst_score
                },
                **entry_exit_prices,
                'risk_level': self._determine_risk_level(signal_strength, confidence, technical_analysis),
                'time_horizon': self._determine_time_horizon(trading_type, catalyst),
                'key_factors': self._identify_key_factors(
                    technical_analysis, fundamental_analysis, sentiment_analysis, weights
                )
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed for {trading_type}: {e}")
            return {
                'signal_type': trading_type,
                'direction': 0,
                'signal_strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_technical_score(self, technical_analysis: Dict, trading_type: str) -> float:
        """Extract normalized technical score for trading type"""
        try:
            if not technical_analysis or 'technical_score' not in technical_analysis:
                return 0.0
            
            tech_scores = technical_analysis.get('technical_score', {})
            
            if trading_type in tech_scores:
                return tech_scores[trading_type]
            
            return tech_scores.get('overall', 0.0)
            
        except Exception:
            return 0.0
    
    def _extract_fundamental_score(self, fundamental_analysis: Dict, trading_type: str) -> float:
        """Extract normalized fundamental score"""
        try:
            if not fundamental_analysis or 'fundamental_score' not in fundamental_analysis:
                return 0.0
            
            fund_score = fundamental_analysis.get('fundamental_score', {})
            overall_score = fund_score.get('overall_score', 0.0)
            
            # Convert 0-1 score to -1 to 1 range for consistency
            # Scores above 0.6 are positive, below 0.4 are negative
            if overall_score > 0.6:
                return (overall_score - 0.6) / 0.4
            elif overall_score < 0.4:
                return (overall_score - 0.4) / 0.4
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_signal_confidence(self, technical_analysis: Dict, fundamental_analysis: Dict,
                                   sentiment_analysis: Dict, catalyst_impact: Dict) -> float:
        """Calculate overall signal confidence"""
        try:
            confidence_factors = []
            
            # Technical analysis confidence
            if technical_analysis and 'data_points' in technical_analysis:
                data_points = technical_analysis['data_points']
                tech_confidence = min(1.0, data_points / 50.0)  # Normalize to 50 data points
                confidence_factors.append(tech_confidence)
            
            # Fundamental analysis confidence
            if fundamental_analysis and 'data_quality' in fundamental_analysis:
                quality_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4, 'unknown': 0.2}
                fund_confidence = quality_map.get(fundamental_analysis['data_quality'], 0.5)
                confidence_factors.append(fund_confidence)
            
            # Sentiment analysis confidence
            if sentiment_analysis and 'confidence' in sentiment_analysis:
                sent_confidence = sentiment_analysis['confidence']
                confidence_factors.append(sent_confidence)
            
            # Catalyst impact confidence
            if catalyst_impact and 'confidence' in catalyst_impact:
                cat_confidence = catalyst_impact['confidence']
                confidence_factors.append(cat_confidence)
            
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5  # Default moderate confidence
                
        except Exception:
            return 0.5
    
    def _calculate_entry_exit_prices(self, technical_analysis: Dict, 
                                   direction: int, trading_type: str) -> Dict[str, float]:
        """Calculate suggested entry, target, and stop-loss prices"""
        try:
            if not technical_analysis or 'last_price' not in technical_analysis:
                return {'entry_price': None, 'target_price': None, 'stop_loss_price': None}
            
            current_price = technical_analysis['last_price']
            atr = technical_analysis.get('indicators', {}).get('ATR', current_price * 0.02)  # Default 2% ATR
            
            if direction == 0:  # Neutral
                return {'entry_price': None, 'target_price': None, 'stop_loss_price': None}
            
            # Risk/reward ratios by trading type
            risk_reward_ratios = {
                'day_trading': {'risk': 1.5, 'reward': 3.0},
                'momentum_trading': {'risk': 2.0, 'reward': 4.0},
                'long_term': {'risk': 3.0, 'reward': 9.0}
            }
            
            ratios = risk_reward_ratios.get(trading_type, risk_reward_ratios['momentum_trading'])
            
            if direction == 1:  # Bullish
                entry_price = current_price
                stop_loss_price = current_price - (atr * ratios['risk'])
                target_price = current_price + (atr * ratios['reward'])
            else:  # Bearish (direction == -1)
                entry_price = current_price
                stop_loss_price = current_price + (atr * ratios['risk'])
                target_price = current_price - (atr * ratios['reward'])
            
            return {
                'entry_price': round(entry_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss_price': round(stop_loss_price, 2)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate entry/exit prices: {e}")
            return {'entry_price': None, 'target_price': None, 'stop_loss_price': None}
    
    def _determine_risk_level(self, signal_strength: float, confidence: float, 
                             technical_analysis: Dict) -> str:
        """Determine risk level for the signal"""
        try:
            # Base risk calculation
            base_risk = 1.0 - (signal_strength * confidence)
            
            # Adjust for volatility
            volatility = technical_analysis.get('indicators', {}).get('volatility_percentile', False)
            if volatility:
                base_risk *= 1.2
            
            # Adjust for unusual volume
            unusual_volume = technical_analysis.get('indicators', {}).get('unusual_volume', False)
            if unusual_volume:
                base_risk *= 1.1
            
            if base_risk < 0.3:
                return 'LOW'
            elif base_risk < 0.6:
                return 'MEDIUM'
            else:
                return 'HIGH'
                
        except Exception:
            return 'MEDIUM'
    
    def _determine_time_horizon(self, trading_type: str, catalyst: Optional[Catalyst] = None) -> str:
        """Determine appropriate time horizon"""
        if trading_type == 'day_trading':
            return '1-3 days'
        elif trading_type == 'momentum_trading':
            if catalyst and catalyst.impact >= 80:
                return '3-14 days'
            else:
                return '1-8 weeks'
        else:  # long_term
            return '3-12 months'
    
    def _estimate_time_horizon(self, catalyst: Catalyst) -> str:
        """Estimate time horizon for catalyst impact"""
        if catalyst.category == CatalystType.EARNINGS:
            return '1-5 days'
        elif catalyst.category == CatalystType.FDA_APPROVAL:
            return '1-30 days'
        elif catalyst.category == CatalystType.MERGER_ACQUISITION:
            return '30-180 days'
        elif catalyst.category == CatalystType.ANALYST:
            return '1-14 days'
        else:
            return '1-30 days'
    
    def _calculate_sentiment_trend(self, catalysts: List[Catalyst]) -> float:
        """Calculate sentiment trend over time"""
        if len(catalysts) < 2:
            return 0.0
        
        try:
            # Sort by date
            sorted_catalysts = sorted(catalysts, key=lambda x: x.published_date or datetime.min)
            
            # Calculate trend using linear regression
            sentiments = [cat.sentiment_score for cat in sorted_catalysts if cat.sentiment_score is not None]
            
            if len(sentiments) < 2:
                return 0.0
            
            x = np.arange(len(sentiments))
            slope, _ = np.polyfit(x, sentiments, 1)
            
            return float(slope)
            
        except Exception:
            return 0.0
    
    def _identify_key_factors(self, technical_analysis: Dict, fundamental_analysis: Dict,
                             sentiment_analysis: Dict, weights: Dict) -> List[str]:
        """Identify key factors driving the signal"""
        factors = []
        
        try:
            # Technical factors
            if technical_analysis.get('patterns', {}):
                patterns = technical_analysis['patterns']
                if patterns.get('bb_breakout'):
                    factors.append(f"Bollinger Band {patterns['bb_breakout']} breakout")
                if patterns.get('volume_spike'):
                    factors.append("Unusual volume spike")
                if patterns.get('macd_bullish_reversal'):
                    factors.append("MACD bullish crossover")
                elif patterns.get('macd_bearish_reversal'):
                    factors.append("MACD bearish crossover")
            
            # Fundamental factors
            if fundamental_analysis.get('investment_signals', {}):
                inv_signals = fundamental_analysis['investment_signals']
                strengths = inv_signals.get('key_strengths', [])
                factors.extend(strengths[:2])  # Top 2 strengths
            
            # Sentiment factors
            if sentiment_analysis.get('catalyst_count', 0) > 5:
                factors.append("High catalyst activity")
            
            return factors[:4]  # Return top 4 factors
            
        except Exception:
            return ["Analysis completed"]
    
    async def _generate_overall_signal(self, ai_signals: Dict, technical_analysis: Dict,
                                     fundamental_analysis: Dict) -> Dict[str, Any]:
        """Generate overall recommendation combining all trading types"""
        try:
            # Weight different trading types based on market conditions
            type_weights = {'day_trading': 0.3, 'momentum_trading': 0.4, 'long_term': 0.3}
            
            # Adjust weights based on volatility
            if technical_analysis.get('indicators', {}).get('volatility_percentile', False):
                # High volatility favors day trading
                type_weights = {'day_trading': 0.5, 'momentum_trading': 0.3, 'long_term': 0.2}
            
            # Calculate weighted overall signal
            overall_direction = 0
            overall_strength = 0.0
            overall_confidence = 0.0
            
            for trading_type, weight in type_weights.items():
                if trading_type in ai_signals:
                    signal = ai_signals[trading_type]
                    overall_direction += signal.get('direction', 0) * weight
                    overall_strength += signal.get('signal_strength', 0) * weight
                    overall_confidence += signal.get('confidence', 0) * weight
            
            # Normalize direction
            if overall_direction > 0.33:
                final_direction = 1
                recommendation = 'BUY'
            elif overall_direction < -0.33:
                final_direction = -1
                recommendation = 'SELL'
            else:
                final_direction = 0
                recommendation = 'HOLD'
            
            return {
                'recommendation': recommendation,
                'direction': final_direction,
                'signal_strength': overall_strength,
                'confidence': overall_confidence,
                'reasoning': self._generate_recommendation_reasoning(ai_signals),
                'best_trading_type': max(ai_signals.keys(), 
                                       key=lambda x: ai_signals[x].get('signal_strength', 0) * 
                                                   ai_signals[x].get('confidence', 0))
            }
            
        except Exception as e:
            logger.error(f"Overall signal generation failed: {e}")
            return {
                'recommendation': 'HOLD',
                'direction': 0,
                'signal_strength': 0.0,
                'confidence': 0.0,
                'reasoning': 'Analysis incomplete',
                'error': str(e)
            }
    
    def _generate_recommendation_reasoning(self, ai_signals: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        try:
            reasoning_parts = []
            
            for trading_type, signal in ai_signals.items():
                if trading_type == 'overall':
                    continue
                    
                strength = signal.get('signal_strength', 0)
                confidence = signal.get('confidence', 0)
                direction = signal.get('direction', 0)
                
                if strength > 0.6 and confidence > 0.7:
                    direction_text = 'bullish' if direction > 0 else 'bearish' if direction < 0 else 'neutral'
                    reasoning_parts.append(
                        f"{trading_type.replace('_', ' ').title()} signals are {direction_text} "
                        f"(strength: {strength:.1f}, confidence: {confidence:.1f})"
                    )
            
            if reasoning_parts:
                return '. '.join(reasoning_parts)
            else:
                return "Mixed signals with moderate confidence across timeframes"
                
        except Exception:
            return "Signal analysis completed"
    
    async def _apply_risk_adjustments(self, ai_signals: Dict, technical_analysis: Dict,
                                    fundamental_analysis: Dict) -> Dict[str, Any]:
        """Apply risk adjustments to signals"""
        try:
            # Get risk factors
            market_cap = fundamental_analysis.get('market_cap', 0)
            sector = fundamental_analysis.get('sector', 'Unknown')
            volatility = technical_analysis.get('indicators', {}).get('volatility_percentile', False)
            
            # Determine risk multipliers
            cap_multiplier = self._get_market_cap_risk_multiplier(market_cap)
            sector_multiplier = self.risk_factors['sector'].get(sector, 1.0)
            vol_multiplier = self.risk_factors['volatility']['high' if volatility else 'medium']
            
            # Apply risk adjustments to each signal
            risk_adjusted_signals = {}
            
            for signal_type, signal in ai_signals.items():
                if isinstance(signal, dict) and 'signal_strength' in signal:
                    adjusted_signal = signal.copy()
                    
                    # Adjust signal strength based on risk
                    original_strength = signal['signal_strength']
                    risk_adjustment = cap_multiplier * sector_multiplier * vol_multiplier
                    
                    # Higher risk = lower effective signal strength
                    adjusted_signal['signal_strength'] = original_strength / risk_adjustment
                    adjusted_signal['risk_adjustment_factor'] = risk_adjustment
                    
                    # Adjust risk level
                    if risk_adjustment > 1.3:
                        adjusted_signal['risk_level'] = 'HIGH'
                    elif risk_adjustment > 1.1:
                        adjusted_signal['risk_level'] = 'MEDIUM'
                    else:
                        adjusted_signal['risk_level'] = 'LOW'
                    
                    risk_adjusted_signals[signal_type] = adjusted_signal
                else:
                    risk_adjusted_signals[signal_type] = signal
            
            return risk_adjusted_signals
            
        except Exception as e:
            logger.error(f"Risk adjustment failed: {e}")
            return ai_signals
    
    def _get_market_cap_risk_multiplier(self, market_cap: float) -> float:
        """Get risk multiplier based on market cap"""
        if market_cap > 200e9:  # > $200B
            return self.risk_factors['market_cap']['mega_cap']
        elif market_cap > 10e9:  # > $10B
            return self.risk_factors['market_cap']['large_cap']
        elif market_cap > 2e9:   # > $2B
            return self.risk_factors['market_cap']['mid_cap']
        elif market_cap > 300e6: # > $300M
            return self.risk_factors['market_cap']['small_cap']
        else:
            return self.risk_factors['market_cap']['micro_cap']
    
    async def _store_analyses(self, ticker: str, technical_analysis: Dict, fundamental_analysis: Dict):
        """Store analysis results in database"""
        try:
            # Store technical analysis
            if technical_analysis:
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.db_manager.store_technical_analysis,
                    technical_analysis
                )
            
            # Store fundamental analysis
            if fundamental_analysis:
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.db_manager.store_fundamental_analysis,
                    fundamental_analysis
                )
            
        except Exception as e:
            logger.warning(f"Failed to store analyses for {ticker}: {e}")
    
    async def _generate_final_recommendations(self, ticker: str, risk_adjusted_signals: Dict,
                                            technical_analysis: Dict, fundamental_analysis: Dict,
                                            catalyst: Optional[Catalyst] = None) -> List[Dict[str, Any]]:
        """Generate final trading recommendations"""
        try:
            recommendations = []
            
            for trading_type, signal in risk_adjusted_signals.items():
                if trading_type == 'overall' or not isinstance(signal, dict):
                    continue
                
                if signal.get('signal_strength', 0) > 0.4 and signal.get('confidence', 0) > 0.6:
                    rec = {
                        'ticker': ticker,
                        'signal_type': trading_type,
                        'recommendation': 'BUY' if signal.get('direction', 0) > 0 else 'SELL',
                        'signal_strength': signal.get('signal_strength', 0),
                        'confidence': signal.get('confidence', 0),
                        'entry_price': signal.get('entry_price'),
                        'target_price': signal.get('target_price'),
                        'stop_loss_price': signal.get('stop_loss_price'),
                        'risk_level': signal.get('risk_level', 'MEDIUM'),
                        'time_horizon': signal.get('time_horizon', 'Unknown'),
                        'key_factors': signal.get('key_factors', []),
                        'catalyst_driven': catalyst is not None,
                        'expires_date': self._calculate_expiry_date(trading_type),
                        'metadata': {
                            'component_scores': signal.get('component_scores', {}),
                            'risk_adjustment_factor': signal.get('risk_adjustment_factor', 1.0),
                            'generated_at': datetime.now().isoformat()
                        }
                    }
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Final recommendation generation failed: {e}")
            return []
    
    def _calculate_expiry_date(self, trading_type: str) -> datetime:
        """Calculate when signal expires"""
        if trading_type == 'day_trading':
            return datetime.now() + timedelta(days=3)
        elif trading_type == 'momentum_trading':
            return datetime.now() + timedelta(weeks=2)
        else:  # long_term
            return datetime.now() + timedelta(weeks=12)
    
    def _calculate_data_quality_score(self, technical_analysis: Dict, 
                                    fundamental_analysis: Dict, sentiment_analysis: Dict) -> float:
        """Calculate overall data quality score"""
        try:
            quality_scores = []
            
            # Technical data quality
            if technical_analysis and 'data_points' in technical_analysis:
                tech_quality = min(1.0, technical_analysis['data_points'] / 50.0)
                quality_scores.append(tech_quality)
            
            # Fundamental data quality
            if fundamental_analysis and 'data_quality' in fundamental_analysis:
                quality_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
                fund_quality = quality_map.get(fundamental_analysis['data_quality'], 0.5)
                quality_scores.append(fund_quality)
            
            # Sentiment data quality
            if sentiment_analysis and 'catalyst_count' in sentiment_analysis:
                sent_quality = min(1.0, sentiment_analysis['catalyst_count'] / 10.0)
                quality_scores.append(sent_quality)
            
            return np.mean(quality_scores) if quality_scores else 0.5
            
        except Exception:
            return 0.5
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.technical_analyzer.cleanup()
            await self.fundamental_analyzer.cleanup()
            self._executor.shutdown(wait=True)
            logger.info("AI Trading Signal Generator cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
