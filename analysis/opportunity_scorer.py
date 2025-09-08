import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import threading

from shared.models import Catalyst, CatalystType
from core.config import AppConfig
from core.database import DatabaseManager
from analysis.signal_generator import AITradingSignalGenerator

logger = logging.getLogger(__name__)

@dataclass
class TradingOpportunity:
    """Trading opportunity data structure"""
    ticker: str
    opportunity_score: float
    confidence: float
    trading_type: str  # day_trading, momentum_trading, long_term
    signal_strength: float
    direction: int  # -1, 0, 1
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss_price: Optional[float]
    risk_level: str
    time_horizon: str
    catalyst_driven: bool
    catalyst_impact: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    key_factors: List[str]
    estimated_return: Optional[float]
    risk_reward_ratio: Optional[float]
    market_cap_category: str
    sector: str
    volatility_level: str
    volume_profile: str
    expires_at: datetime
    metadata: Dict[str, Any]

class CatalystOpportunityScorer:
    """Advanced opportunity scorer for catalyst-driven trading opportunities"""
    
    def __init__(self, config: AppConfig, db_manager: DatabaseManager, 
                 signal_generator: AITradingSignalGenerator):
        self.config = config
        self.db_manager = db_manager
        self.signal_generator = signal_generator
        
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Opportunity scoring weights
        self.scoring_weights = {
            'signal_strength': 0.25,
            'confidence': 0.20,
            'catalyst_impact': 0.20,
            'risk_reward_ratio': 0.15,
            'technical_momentum': 0.10,
            'market_conditions': 0.10
        }
        
        # Market condition factors
        self.market_factors = {
            'high_volatility_bonus': 0.1,
            'unusual_volume_bonus': 0.05,
            'sector_strength_bonus': 0.05,
            'earnings_season_bonus': 0.03
        }
        
        # Risk penalties
        self.risk_penalties = {
            'low_confidence': -0.15,
            'high_risk_sector': -0.10,
            'micro_cap_penalty': -0.08,
            'poor_liquidity': -0.05
        }
        
        logger.info("Catalyst Opportunity Scorer initialized")
    
    async def scan_for_opportunities(self, 
                                   catalyst_filters: Optional[Dict] = None,
                                   min_opportunity_score: float = 0.6,
                                   max_results: int = 20) -> List[TradingOpportunity]:
        """Scan for high-probability trading opportunities"""
        try:
            logger.info("Scanning for catalyst-driven trading opportunities")
            
            # Get recent high-impact catalysts
            catalysts = self._get_filtered_catalysts(catalyst_filters or {})
            
            if not catalysts:
                logger.info("No catalysts found matching criteria")
                return []
            
            # Process catalysts concurrently for efficiency
            opportunity_tasks = []
            
            for catalyst in catalysts[:50]:  # Limit to top 50 catalysts
                task = self._evaluate_catalyst_opportunity(catalyst)
                opportunity_tasks.append(task)
            
            # Execute all evaluations concurrently
            opportunity_results = await asyncio.gather(
                *opportunity_tasks, return_exceptions=True
            )
            
            # Filter and rank opportunities
            valid_opportunities = []
            
            for result in opportunity_results:
                if isinstance(result, Exception):
                    logger.warning(f"Opportunity evaluation failed: {result}")
                    continue
                
                if result and result.opportunity_score >= min_opportunity_score:
                    valid_opportunities.append(result)
            
            # Sort by opportunity score (highest first)
            valid_opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Apply diversification and final filtering
            final_opportunities = self._apply_portfolio_diversification(
                valid_opportunities, max_results
            )
            
            logger.info(f"Found {len(final_opportunities)} high-probability opportunities")
            return final_opportunities
            
        except Exception as e:
            logger.error(f"Error scanning for opportunities: {e}")
            return []
    
    def _get_filtered_catalysts(self, filters: Dict) -> List[Catalyst]:
        """Get catalysts matching specified filters"""
        try:
            # Default filters for high-quality catalysts
            default_filters = {
                'min_impact': filters.get('min_impact', 60),
                'categories': filters.get('categories', None),
                'max_age_hours': filters.get('max_age_hours', 168),  # 1 week
                'min_confidence': filters.get('min_confidence', 0.6),
                'sectors': filters.get('sectors', None)
            }
            
            # Get recent catalysts from database
            all_catalysts = self.db_manager.get_catalysts(
                min_impact=default_filters['min_impact'],
                limit=200  # Get larger pool for filtering
            )
            
            filtered_catalysts = []
            
            for catalyst in all_catalysts:
                # Age filter
                if catalyst.published_date:
                    age_hours = (datetime.now() - catalyst.published_date).total_seconds() / 3600
                    if age_hours > default_filters['max_age_hours']:
                        continue
                
                # Confidence filter
                if catalyst.confidence < default_filters['min_confidence']:
                    continue
                
                # Category filter
                if default_filters['categories']:
                    if catalyst.category not in default_filters['categories']:
                        continue
                
                # Sector filter
                if default_filters['sectors']:
                    if catalyst.sector not in default_filters['sectors']:
                        continue
                
                filtered_catalysts.append(catalyst)
            
            return filtered_catalysts
            
        except Exception as e:
            logger.error(f"Error filtering catalysts: {e}")
            return []
    
    async def _evaluate_catalyst_opportunity(self, catalyst: Catalyst) -> Optional[TradingOpportunity]:
        """Evaluate a single catalyst for trading opportunity"""
        try:
            # Generate comprehensive signals for the catalyst's ticker
            signals_data = await self.signal_generator.generate_comprehensive_signals(
                catalyst.ticker, catalyst=catalyst
            )
            
            if 'error' in signals_data:
                logger.warning(f"Signal generation failed for {catalyst.ticker}: {signals_data['error']}")
                return None
            
            # Extract analysis data
            ai_signals = signals_data.get('ai_signals', {})
            technical_analysis = signals_data.get('technical_analysis', {})
            fundamental_analysis = signals_data.get('fundamental_analysis', {})
            catalyst_impact = signals_data.get('catalyst_impact', {})
            
            # Find the best trading signal
            best_signal = self._find_best_trading_signal(ai_signals)
            
            if not best_signal or best_signal['signal_strength'] < 0.4:
                return None  # Signal too weak
            
            # Calculate opportunity score
            opportunity_score = await self._calculate_opportunity_score(
                best_signal, catalyst, technical_analysis, 
                fundamental_analysis, catalyst_impact
            )
            
            # Extract market context
            market_context = self._extract_market_context(
                technical_analysis, fundamental_analysis
            )
            
            # Calculate estimated return and risk metrics
            return_metrics = self._calculate_return_metrics(best_signal)
            
            # Create trading opportunity
            opportunity = TradingOpportunity(
                ticker=catalyst.ticker,
                opportunity_score=opportunity_score,
                confidence=best_signal.get('confidence', 0.0),
                trading_type=best_signal.get('signal_type', 'momentum_trading'),
                signal_strength=best_signal.get('signal_strength', 0.0),
                direction=best_signal.get('direction', 0),
                entry_price=best_signal.get('entry_price'),
                target_price=best_signal.get('target_price'),
                stop_loss_price=best_signal.get('stop_loss_price'),
                risk_level=best_signal.get('risk_level', 'MEDIUM'),
                time_horizon=best_signal.get('time_horizon', 'Unknown'),
                catalyst_driven=True,
                catalyst_impact=catalyst_impact.get('impact_score', 0.0),
                technical_score=technical_analysis.get('technical_score', {}).get('overall', 0.0),
                fundamental_score=fundamental_analysis.get('fundamental_score', {}).get('overall_score', 0.0),
                sentiment_score=signals_data.get('sentiment_analysis', {}).get('overall_sentiment', 0.0),
                key_factors=best_signal.get('key_factors', []),
                estimated_return=return_metrics.get('estimated_return'),
                risk_reward_ratio=return_metrics.get('risk_reward_ratio'),
                market_cap_category=market_context['market_cap_category'],
                sector=market_context['sector'],
                volatility_level=market_context['volatility_level'],
                volume_profile=market_context['volume_profile'],
                expires_at=self._calculate_opportunity_expiry(best_signal),
                metadata={
                    'catalyst_id': catalyst.id if hasattr(catalyst, 'id') else None,
                    'catalyst_category': catalyst.category.value if catalyst.category else None,
                    'catalyst_confidence': catalyst.confidence,
                    'catalyst_impact': catalyst.impact,
                    'component_scores': best_signal.get('component_scores', {}),
                    'data_quality_score': signals_data.get('data_quality_score', 0.5),
                    'generated_at': datetime.now().isoformat()
                }
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error evaluating opportunity for {catalyst.ticker}: {e}")
            return None
    
    def _find_best_trading_signal(self, ai_signals: Dict) -> Optional[Dict]:
        """Find the best trading signal from AI signals"""
        try:
            best_signal = None
            best_score = 0.0
            
            for signal_type, signal in ai_signals.items():
                if signal_type == 'overall' or not isinstance(signal, dict):
                    continue
                
                # Calculate combined score (strength * confidence)
                combined_score = (
                    signal.get('signal_strength', 0.0) * 
                    signal.get('confidence', 0.0)
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_signal = signal
            
            return best_signal
            
        except Exception:
            return None
    
    async def _calculate_opportunity_score(self, signal: Dict, catalyst: Catalyst,
                                         technical_analysis: Dict, fundamental_analysis: Dict,
                                         catalyst_impact: Dict) -> float:
        """Calculate comprehensive opportunity score"""
        try:
            # Base signal metrics
            signal_strength = signal.get('signal_strength', 0.0)
            confidence = signal.get('confidence', 0.0)
            catalyst_impact_score = catalyst_impact.get('impact_score', 0.0)
            
            # Technical momentum
            technical_momentum = self._calculate_technical_momentum(technical_analysis)
            
            # Risk-reward ratio
            risk_reward = self._calculate_risk_reward_ratio(signal)
            
            # Market condition factor
            market_condition_factor = self._calculate_market_condition_factor(
                technical_analysis, fundamental_analysis, catalyst
            )
            
            # Calculate base opportunity score
            base_score = (
                signal_strength * self.scoring_weights['signal_strength'] +
                confidence * self.scoring_weights['confidence'] +
                catalyst_impact_score * self.scoring_weights['catalyst_impact'] +
                risk_reward * self.scoring_weights['risk_reward_ratio'] +
                technical_momentum * self.scoring_weights['technical_momentum'] +
                market_condition_factor * self.scoring_weights['market_conditions']
            )
            
            # Apply bonuses and penalties
            final_score = self._apply_scoring_adjustments(
                base_score, signal, technical_analysis, fundamental_analysis, catalyst
            )
            
            return min(1.0, max(0.0, final_score))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.0
    
    def _calculate_technical_momentum(self, technical_analysis: Dict) -> float:
        """Calculate technical momentum score"""
        try:
            if not technical_analysis:
                return 0.0
            
            momentum_factors = []
            
            # MACD momentum
            patterns = technical_analysis.get('patterns', {})
            if patterns.get('macd_bullish_reversal'):
                momentum_factors.append(0.8)
            elif patterns.get('macd_bearish_reversal'):
                momentum_factors.append(-0.8)
            
            # Moving average alignment
            if patterns.get('ma_alignment') == 'bullish':
                momentum_factors.append(0.7)
            elif patterns.get('ma_alignment') == 'bearish':
                momentum_factors.append(-0.7)
            
            # Volume confirmation
            if patterns.get('volume_spike'):
                momentum_factors.append(0.6)
            
            # Breakout patterns
            if patterns.get('bb_breakout') == 'bullish':
                momentum_factors.append(0.5)
            elif patterns.get('bb_breakout') == 'bearish':
                momentum_factors.append(-0.5)
            
            if momentum_factors:
                return np.mean(momentum_factors)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_risk_reward_ratio(self, signal: Dict) -> float:
        """Calculate normalized risk-reward ratio"""
        try:
            entry_price = signal.get('entry_price')
            target_price = signal.get('target_price')
            stop_loss_price = signal.get('stop_loss_price')
            
            if not all([entry_price, target_price, stop_loss_price]):
                return 0.5  # Neutral if prices not available
            
            direction = signal.get('direction', 0)
            
            if direction == 1:  # Bullish
                potential_gain = target_price - entry_price
                potential_loss = entry_price - stop_loss_price
            elif direction == -1:  # Bearish
                potential_gain = entry_price - target_price
                potential_loss = stop_loss_price - entry_price
            else:
                return 0.0
            
            if potential_loss <= 0:
                return 0.0  # Invalid risk setup
            
            risk_reward_ratio = potential_gain / potential_loss
            
            # Normalize to 0-1 scale (ratios > 3:1 get max score)
            normalized_ratio = min(1.0, risk_reward_ratio / 3.0)
            
            return normalized_ratio
            
        except Exception:
            return 0.5
    
    def _calculate_market_condition_factor(self, technical_analysis: Dict, 
                                         fundamental_analysis: Dict, catalyst: Catalyst) -> float:
        """Calculate market condition factor"""
        try:
            condition_score = 0.5  # Base neutral score
            
            # High volatility can be good for trading
            if technical_analysis.get('indicators', {}).get('volatility_percentile'):
                condition_score += self.market_factors['high_volatility_bonus']
            
            # Unusual volume indicates interest
            if technical_analysis.get('indicators', {}).get('unusual_volume'):
                condition_score += self.market_factors['unusual_volume_bonus']
            
            # Sector strength
            sector = fundamental_analysis.get('sector')
            if sector in ['Technology', 'Healthcare', 'Consumer Cyclical']:
                condition_score += self.market_factors['sector_strength_bonus']
            
            # Earnings season bonus for earnings catalysts
            if catalyst.category == CatalystType.EARNINGS:
                condition_score += self.market_factors['earnings_season_bonus']
            
            return min(1.0, condition_score)
            
        except Exception:
            return 0.5
    
    def _apply_scoring_adjustments(self, base_score: float, signal: Dict,
                                  technical_analysis: Dict, fundamental_analysis: Dict,
                                  catalyst: Catalyst) -> float:
        """Apply bonuses and penalties to base score"""
        try:
            adjusted_score = base_score
            
            # Confidence penalty
            if signal.get('confidence', 0.0) < 0.5:
                adjusted_score += self.risk_penalties['low_confidence']
            
            # Sector risk penalty
            high_risk_sectors = ['Energy', 'Biotechnology']
            if fundamental_analysis.get('sector') in high_risk_sectors:
                adjusted_score += self.risk_penalties['high_risk_sector']
            
            # Market cap penalty
            market_cap = fundamental_analysis.get('market_cap', 0)
            if market_cap < 300e6:  # Micro cap
                adjusted_score += self.risk_penalties['micro_cap_penalty']
            
            # Low liquidity penalty
            volume_ratio = technical_analysis.get('indicators', {}).get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                adjusted_score += self.risk_penalties['poor_liquidity']
            
            return adjusted_score
            
        except Exception:
            return base_score
    
    def _extract_market_context(self, technical_analysis: Dict, 
                              fundamental_analysis: Dict) -> Dict[str, str]:
        """Extract market context information"""
        try:
            # Market cap category
            market_cap = fundamental_analysis.get('market_cap', 0)
            if market_cap > 200e9:
                cap_category = 'Mega Cap'
            elif market_cap > 10e9:
                cap_category = 'Large Cap'
            elif market_cap > 2e9:
                cap_category = 'Mid Cap'
            elif market_cap > 300e6:
                cap_category = 'Small Cap'
            else:
                cap_category = 'Micro Cap'
            
            # Volatility level
            if technical_analysis.get('indicators', {}).get('volatility_percentile'):
                volatility = 'High'
            else:
                volatility = 'Normal'
            
            # Volume profile
            volume_ratio = technical_analysis.get('indicators', {}).get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                volume_profile = 'Unusual High'
            elif volume_ratio > 1.5:
                volume_profile = 'Above Average'
            elif volume_ratio < 0.5:
                volume_profile = 'Below Average'
            else:
                volume_profile = 'Normal'
            
            return {
                'market_cap_category': cap_category,
                'sector': fundamental_analysis.get('sector', 'Unknown'),
                'volatility_level': volatility,
                'volume_profile': volume_profile
            }
            
        except Exception:
            return {
                'market_cap_category': 'Unknown',
                'sector': 'Unknown',
                'volatility_level': 'Normal',
                'volume_profile': 'Normal'
            }
    
    def _calculate_return_metrics(self, signal: Dict) -> Dict[str, Optional[float]]:
        """Calculate estimated return and risk metrics"""
        try:
            entry_price = signal.get('entry_price')
            target_price = signal.get('target_price')
            stop_loss_price = signal.get('stop_loss_price')
            direction = signal.get('direction', 0)
            
            if not all([entry_price, target_price, stop_loss_price]) or direction == 0:
                return {'estimated_return': None, 'risk_reward_ratio': None}
            
            if direction == 1:  # Bullish
                estimated_return = (target_price - entry_price) / entry_price
                risk = (entry_price - stop_loss_price) / entry_price
            else:  # Bearish
                estimated_return = (entry_price - target_price) / entry_price
                risk = (stop_loss_price - entry_price) / entry_price
            
            risk_reward_ratio = estimated_return / risk if risk > 0 else None
            
            return {
                'estimated_return': estimated_return,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception:
            return {'estimated_return': None, 'risk_reward_ratio': None}
    
    def _calculate_opportunity_expiry(self, signal: Dict) -> datetime:
        """Calculate when opportunity expires"""
        try:
            time_horizon = signal.get('time_horizon', '1-8 weeks')
            
            if 'day' in time_horizon:
                days = 3
            elif 'week' in time_horizon:
                days = 14
            elif 'month' in time_horizon:
                days = 90
            else:
                days = 7  # Default
            
            return datetime.now() + timedelta(days=days)
            
        except Exception:
            return datetime.now() + timedelta(days=7)
    
    def _apply_portfolio_diversification(self, opportunities: List[TradingOpportunity],
                                       max_results: int) -> List[TradingOpportunity]:
        """Apply portfolio diversification rules"""
        try:
            if len(opportunities) <= max_results:
                return opportunities
            
            # Group by sector and trading type for diversification
            sector_counts = {}
            trading_type_counts = {}
            selected_opportunities = []
            
            for opportunity in opportunities:
                # Check sector diversification (max 40% in one sector)
                sector_limit = max(1, int(max_results * 0.4))
                sector_count = sector_counts.get(opportunity.sector, 0)
                
                # Check trading type diversification
                type_count = trading_type_counts.get(opportunity.trading_type, 0)
                type_limit = max(1, int(max_results * 0.6))  # Max 60% in one trading type
                
                # Apply diversification rules
                if (sector_count < sector_limit and 
                    type_count < type_limit and 
                    len(selected_opportunities) < max_results):
                    
                    selected_opportunities.append(opportunity)
                    sector_counts[opportunity.sector] = sector_count + 1
                    trading_type_counts[opportunity.trading_type] = type_count + 1
            
            # Fill remaining slots with best opportunities if needed
            remaining_slots = max_results - len(selected_opportunities)
            if remaining_slots > 0:
                unselected = [op for op in opportunities if op not in selected_opportunities]
                selected_opportunities.extend(unselected[:remaining_slots])
            
            return selected_opportunities
            
        except Exception as e:
            logger.error(f"Error applying diversification: {e}")
            return opportunities[:max_results]
    
    async def get_opportunity_by_ticker(self, ticker: str) -> Optional[TradingOpportunity]:
        """Get current opportunity for a specific ticker"""
        try:
            # Get recent catalyst for this ticker
            recent_catalysts = self.db_manager.get_catalysts(ticker=ticker, limit=1)
            
            if not recent_catalysts:
                return None
            
            catalyst = recent_catalysts[0]
            opportunity = await self._evaluate_catalyst_opportunity(catalyst)
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error getting opportunity for {ticker}: {e}")
            return None
    
    def get_opportunity_summary_stats(self, opportunities: List[TradingOpportunity]) -> Dict[str, Any]:
        """Get summary statistics for opportunities"""
        try:
            if not opportunities:
                return {}
            
            # Basic stats
            scores = [op.opportunity_score for op in opportunities]
            confidences = [op.confidence for op in opportunities]
            estimated_returns = [op.estimated_return for op in opportunities if op.estimated_return]
            
            # Sector distribution
            sectors = {}
            for op in opportunities:
                sectors[op.sector] = sectors.get(op.sector, 0) + 1
            
            # Trading type distribution
            trading_types = {}
            for op in opportunities:
                trading_types[op.trading_type] = trading_types.get(op.trading_type, 0) + 1
            
            # Risk level distribution
            risk_levels = {}
            for op in opportunities:
                risk_levels[op.risk_level] = risk_levels.get(op.risk_level, 0) + 1
            
            return {
                'total_opportunities': len(opportunities),
                'avg_opportunity_score': np.mean(scores) if scores else 0.0,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'avg_estimated_return': np.mean(estimated_returns) if estimated_returns else None,
                'sector_distribution': sectors,
                'trading_type_distribution': trading_types,
                'risk_level_distribution': risk_levels,
                'catalyst_driven_count': sum(1 for op in opportunities if op.catalyst_driven),
                'high_confidence_count': sum(1 for op in opportunities if op.confidence > 0.8)
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
