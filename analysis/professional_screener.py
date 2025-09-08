import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ScreenerSignal:
    """Professional stock screener signal"""
    ticker: str
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    catalyst_category: str
    impact_score: float
    confidence: float
    time_horizon: str  # 'short', 'medium', 'long'
    risk_level: str  # 'low', 'medium', 'high'
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None

class ProfessionalStockScreener:
    """Professional-grade stock screening and analysis methods used by institutional investors"""
    
    def __init__(self, config):
        self.config = config
        
        # Professional screening criteria
        self.screening_thresholds = {
            'min_impact_score': 70.0,  # Minimum impact for institutional attention
            'min_confidence': 0.6,     # Minimum confidence for signal generation
            'max_risk_tolerance': 0.8,  # Maximum acceptable risk
            'volume_threshold': 1.5,    # Minimum volume multiple
            'price_momentum_threshold': 0.05,  # 5% price movement threshold
        }
        
        # Institutional-grade analysis weights
        self.analysis_weights = {
            'catalyst_impact': 0.30,
            'sentiment_strength': 0.25,
            'timing_factor': 0.20,
            'market_conditions': 0.15,
            'risk_assessment': 0.10
        }
        
        # Professional categorization mapping
        self.professional_categories = {
            'earnings': {
                'institutional_priority': 'high',
                'typical_impact_duration': 'short_term',
                'risk_factors': ['volatility', 'market_expectations'],
                'analysis_methods': ['eps_beat_analysis', 'guidance_analysis', 'sector_comparison']
            },
            'ma': {
                'institutional_priority': 'very_high',
                'typical_impact_duration': 'long_term',
                'risk_factors': ['regulatory_approval', 'financing', 'integration'],
                'analysis_methods': ['deal_premium_analysis', 'synergy_assessment', 'market_consolidation']
            },
            'regulatory': {
                'institutional_priority': 'high',
                'typical_impact_duration': 'medium_term',
                'risk_factors': ['regulatory_delay', 'approval_conditions', 'market_access'],
                'analysis_methods': ['regulatory_timeline', 'competitive_landscape', 'market_potential']
            },
            'insider_trading': {
                'institutional_priority': 'medium',
                'typical_impact_duration': 'short_term',
                'risk_factors': ['insider_knowledge', 'timing', 'volume'],
                'analysis_methods': ['insider_sentiment', 'transaction_clustering', 'historical_patterns']
            }
        }
        
        logger.info("Professional Stock Screener initialized with institutional-grade analysis methods")
    
    async def analyze_catalyst_opportunity(self, catalyst_data: Dict[str, Any]) -> ScreenerSignal:
        """Comprehensive institutional-grade catalyst analysis"""
        try:
            ticker = catalyst_data.get('ticker', '').upper()
            category = catalyst_data.get('category', 'general')
            
            # Multi-factor analysis
            impact_analysis = await self._analyze_catalyst_impact(catalyst_data)
            sentiment_analysis = await self._analyze_sentiment_strength(catalyst_data)
            timing_analysis = await self._analyze_timing_factors(catalyst_data)
            market_analysis = await self._analyze_market_conditions(catalyst_data)
            risk_analysis = await self._analyze_risk_factors(catalyst_data)
            
            # Calculate composite professional score
            composite_score = self._calculate_composite_score({
                'impact': impact_analysis,
                'sentiment': sentiment_analysis,
                'timing': timing_analysis,
                'market': market_analysis,
                'risk': risk_analysis
            })
            
            # Generate professional signal
            signal = self._generate_professional_signal(
                ticker, category, composite_score, catalyst_data
            )
            
            # Add institutional metadata
            signal.metadata = {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'analysis_components': {
                    'impact_score': impact_analysis['score'],
                    'sentiment_strength': sentiment_analysis['strength'],
                    'timing_score': timing_analysis['score'],
                    'market_score': market_analysis['score'],
                    'risk_score': risk_analysis['score']
                },
                'professional_grade': True,
                'institutional_priority': self.professional_categories.get(category, {}).get('institutional_priority', 'low'),
                'screening_version': '2.0'
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in professional catalyst analysis: {e}")
            return self._create_default_signal(catalyst_data.get('ticker', 'UNKNOWN'))
    
    async def _analyze_catalyst_impact(self, catalyst_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze catalyst impact using institutional methods"""
        try:
            category = catalyst_data.get('category', 'general')
            impact_score = catalyst_data.get('impact', 0)
            
            # Professional impact adjustment factors
            category_multipliers = {
                'earnings': 1.2,  # Earnings have immediate market impact
                'ma': 1.8,        # M&A has highest impact
                'regulatory': 1.5, # Regulatory decisions are high impact
                'insider_trading': 0.8, # Insider trading is lower direct impact
                'general': 1.0
            }
            
            # Adjust impact based on professional category analysis
            adjusted_impact = impact_score * category_multipliers.get(category, 1.0)
            
            # Market cap consideration (larger companies have more stable impact)
            market_cap_factor = self._estimate_market_cap_factor(catalyst_data.get('ticker', ''))
            
            # Final professional impact score
            professional_impact = min(100.0, adjusted_impact * market_cap_factor)
            
            return {
                'score': professional_impact,
                'category_multiplier': category_multipliers.get(category, 1.0),
                'market_cap_factor': market_cap_factor,
                'institutional_relevance': professional_impact >= self.screening_thresholds['min_impact_score']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing catalyst impact: {e}")
            return {'score': 0.0, 'institutional_relevance': False}
    
    async def _analyze_sentiment_strength(self, catalyst_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment strength using professional methods"""
        try:
            sentiment_score = catalyst_data.get('sentiment_score', 0.0)
            sentiment_label = catalyst_data.get('sentiment_label', 'neutral')
            confidence = catalyst_data.get('confidence', 0.0)
            
            # Professional sentiment strength calculation
            if sentiment_label == 'positive':
                base_strength = min(1.0, abs(sentiment_score))
            elif sentiment_label == 'negative':
                base_strength = min(1.0, abs(sentiment_score)) * 0.9  # Negative news slightly discounted
            else:
                base_strength = 0.3  # Neutral sentiment has lower strength
            
            # Confidence adjustment (institutional traders require high confidence)
            confidence_adjusted_strength = base_strength * confidence
            
            # Source credibility factor
            source = catalyst_data.get('source', 'unknown')
            source_credibility = self._get_source_credibility(source)
            
            final_strength = confidence_adjusted_strength * source_credibility
            
            return {
                'strength': final_strength,
                'base_sentiment': base_strength,
                'confidence_factor': confidence,
                'source_credibility': source_credibility,
                'institutional_grade': final_strength >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment strength: {e}")
            return {'strength': 0.0, 'institutional_grade': False}
    
    async def _analyze_timing_factors(self, catalyst_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing factors for institutional trading"""
        try:
            published_date = catalyst_data.get('published_date')
            category = catalyst_data.get('category', 'general')
            
            # Calculate recency factor
            if published_date:
                if isinstance(published_date, str):
                    pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                else:
                    pub_date = published_date
                
                time_diff = datetime.now(timezone.utc) - pub_date
                hours_old = time_diff.total_seconds() / 3600
                
                # Professional timing scoring
                if hours_old <= 1:
                    recency_score = 1.0  # Breaking news
                elif hours_old <= 6:
                    recency_score = 0.9  # Very fresh
                elif hours_old <= 24:
                    recency_score = 0.7  # Recent
                elif hours_old <= 72:
                    recency_score = 0.5  # Acceptable
                else:
                    recency_score = 0.2  # Too old for institutional action
            else:
                recency_score = 0.3
            
            # Market timing factor (trading hours, market open/close)
            market_timing_score = self._calculate_market_timing_score()
            
            # Category-specific timing analysis
            category_timing = self.professional_categories.get(category, {})
            impact_duration = category_timing.get('typical_impact_duration', 'short_term')
            
            # Duration adjustment for institutional strategies
            duration_scores = {
                'short_term': 0.8,   # Quick institutional plays
                'medium_term': 1.0,  # Ideal for institutional positioning
                'long_term': 0.9     # Strategic institutional investments
            }
            
            duration_score = duration_scores.get(impact_duration, 0.7)
            
            # Composite timing score
            composite_timing = (recency_score * 0.5 + market_timing_score * 0.3 + duration_score * 0.2)
            
            return {
                'score': composite_timing,
                'recency_score': recency_score,
                'market_timing': market_timing_score,
                'duration_fit': duration_score,
                'hours_old': hours_old if published_date else None,
                'institutional_timing': composite_timing >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timing factors: {e}")
            return {'score': 0.0, 'institutional_timing': False}
    
    async def _analyze_market_conditions(self, catalyst_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions for institutional decision making"""
        try:
            # Market volatility assessment
            volatility_score = self._assess_market_volatility()
            
            # Sector performance analysis
            sector = catalyst_data.get('sector', 'Unknown')
            sector_score = self._assess_sector_performance(sector)
            
            # Market liquidity assessment
            ticker = catalyst_data.get('ticker', '')
            liquidity_score = self._assess_stock_liquidity(ticker)
            
            # Overall market sentiment (VIX proxy)
            market_sentiment = self._assess_overall_market_sentiment()
            
            # Composite market conditions score
            market_score = (
                volatility_score * 0.3 +
                sector_score * 0.25 +
                liquidity_score * 0.25 +
                market_sentiment * 0.2
            )
            
            return {
                'score': market_score,
                'volatility': volatility_score,
                'sector_performance': sector_score,
                'liquidity': liquidity_score,
                'market_sentiment': market_sentiment,
                'institutional_favorable': market_score >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'score': 0.5, 'institutional_favorable': False}
    
    async def _analyze_risk_factors(self, catalyst_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional risk assessment for institutional trading"""
        try:
            category = catalyst_data.get('category', 'general')
            impact_score = catalyst_data.get('impact', 0)
            confidence = catalyst_data.get('confidence', 0.0)
            
            # Category-specific risk factors
            category_risks = self.professional_categories.get(category, {}).get('risk_factors', [])
            
            # Base risk calculation
            base_risk = 1.0 - confidence  # Lower confidence = higher risk
            
            # Impact-based risk (higher impact can mean higher volatility)
            impact_risk = min(0.5, impact_score / 200.0)  # Scale impact to risk
            
            # Category risk multipliers
            category_risk_multipliers = {
                'earnings': 0.7,  # Earnings are predictable
                'ma': 0.9,        # M&A has execution risk
                'regulatory': 0.8, # Regulatory has timing risk
                'insider_trading': 0.6, # Insider trading has lower direct risk
                'general': 0.8
            }
            
            category_risk = category_risk_multipliers.get(category, 0.8)
            
            # Calculate composite risk score
            composite_risk = min(1.0, (base_risk + impact_risk) * category_risk)
            
            # Risk level classification
            if composite_risk <= 0.3:
                risk_level = 'low'
            elif composite_risk <= 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'score': 1.0 - composite_risk,  # Higher score = lower risk
                'risk_level': risk_level,
                'base_risk': base_risk,
                'impact_risk': impact_risk,
                'category_risk': category_risk,
                'institutional_acceptable': composite_risk <= self.screening_thresholds['max_risk_tolerance']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {e}")
            return {'score': 0.3, 'risk_level': 'high', 'institutional_acceptable': False}
    
    def _calculate_composite_score(self, analysis_components: Dict[str, Dict]) -> float:
        """Calculate weighted composite score for institutional decision making"""
        try:
            weighted_score = 0.0
            
            for component, weight in self.analysis_weights.items():
                if component == 'catalyst_impact':
                    weighted_score += analysis_components['impact']['score'] / 100.0 * weight
                elif component == 'sentiment_strength':
                    weighted_score += analysis_components['sentiment']['strength'] * weight
                elif component == 'timing_factor':
                    weighted_score += analysis_components['timing']['score'] * weight
                elif component == 'market_conditions':
                    weighted_score += analysis_components['market']['score'] * weight
                elif component == 'risk_assessment':
                    weighted_score += analysis_components['risk']['score'] * weight
            
            return min(1.0, weighted_score)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.0
    
    def _generate_professional_signal(self, ticker: str, category: str, 
                                    composite_score: float, catalyst_data: Dict) -> ScreenerSignal:
        """Generate professional trading signal"""
        try:
            # Signal type determination
            sentiment = catalyst_data.get('sentiment_label', 'neutral')
            impact = catalyst_data.get('impact', 0)
            
            if composite_score >= 0.7 and sentiment == 'positive':
                signal_type = 'bullish'
            elif composite_score >= 0.7 and sentiment == 'negative':
                signal_type = 'bearish'
            elif composite_score >= 0.5:
                signal_type = 'neutral'
            else:
                signal_type = 'neutral'
            
            # Strength calculation
            strength = composite_score
            
            # Time horizon based on category
            category_horizons = {
                'earnings': 'short',     # 1-3 days
                'ma': 'long',           # 3-12 months
                'regulatory': 'medium',  # 1-6 months
                'insider_trading': 'short', # 1-7 days
                'general': 'medium'
            }
            
            time_horizon = category_horizons.get(category, 'medium')
            
            # Risk level from analysis
            risk_analysis = catalyst_data.get('risk_analysis', {})
            risk_level = risk_analysis.get('risk_level', 'medium')
            
            return ScreenerSignal(
                ticker=ticker,
                signal_type=signal_type,
                strength=strength,
                catalyst_category=category,
                impact_score=impact,
                confidence=composite_score,
                time_horizon=time_horizon,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error generating professional signal: {e}")
            return self._create_default_signal(ticker)
    
    def _create_default_signal(self, ticker: str) -> ScreenerSignal:
        """Create default signal for error cases"""
        return ScreenerSignal(
            ticker=ticker,
            signal_type='neutral',
            strength=0.0,
            catalyst_category='general',
            impact_score=0.0,
            confidence=0.0,
            time_horizon='medium',
            risk_level='high'
        )
    
    # Helper methods for professional analysis
    def _estimate_market_cap_factor(self, ticker: str) -> float:
        """Estimate market cap factor (would integrate with real data in production)"""
        # Large cap stocks (professional focus) get higher factors
        large_cap_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        if ticker in large_cap_tickers:
            return 1.2
        return 1.0
    
    def _get_source_credibility(self, source: str) -> float:
        """Get source credibility factor for institutional use"""
        credible_sources = {
            'Reuters': 1.0,
            'Bloomberg': 1.0,
            'SEC EDGAR': 0.95,
            'FDA': 0.95,
            'NewsAPI': 0.8,
            'Yahoo Finance': 0.7,
            'MarketWatch': 0.7
        }
        return credible_sources.get(source, 0.5)
    
    def _calculate_market_timing_score(self) -> float:
        """Calculate market timing score based on trading hours"""
        now = datetime.now()
        hour = now.hour
        
        # US market hours: 9:30 AM - 4:00 PM EST
        if 9 <= hour <= 16:
            return 1.0  # Market hours
        elif 7 <= hour <= 9 or 16 <= hour <= 18:
            return 0.8  # Pre/post market
        else:
            return 0.4  # Off hours
    
    def _assess_market_volatility(self) -> float:
        """Assess current market volatility (placeholder for VIX integration)"""
        # In production, this would integrate with VIX or other volatility indices
        return 0.7  # Moderate volatility assumption
    
    def _assess_sector_performance(self, sector: str) -> float:
        """Assess sector performance (placeholder for sector analysis)"""
        # In production, this would analyze sector ETF performance
        return 0.6  # Neutral sector performance assumption
    
    def _assess_stock_liquidity(self, ticker: str) -> float:
        """Assess stock liquidity (placeholder for volume analysis)"""
        # In production, this would analyze average daily volume
        return 0.8  # High liquidity assumption for most stocks
    
    def _assess_overall_market_sentiment(self) -> float:
        """Assess overall market sentiment (placeholder for market indicators)"""
        # In production, this would integrate with market sentiment indicators
        return 0.6  # Neutral market sentiment assumption
    
    async def screen_multiple_catalysts(self, catalyst_list: List[Dict[str, Any]]) -> List[ScreenerSignal]:
        """Screen multiple catalysts using professional methods"""
        try:
            signals = []
            
            # Process each catalyst with professional analysis
            for catalyst_data in catalyst_list:
                signal = await self.analyze_catalyst_opportunity(catalyst_data)
                
                # Apply institutional filters
                if self._passes_institutional_filters(signal):
                    signals.append(signal)
            
            # Sort by professional priority
            signals.sort(key=lambda x: (x.strength, x.impact_score), reverse=True)
            
            logger.info(f"Professional screening: {len(signals)} signals generated from {len(catalyst_list)} catalysts")
            return signals
            
        except Exception as e:
            logger.error(f"Error in professional catalyst screening: {e}")
            return []
    
    def _passes_institutional_filters(self, signal: ScreenerSignal) -> bool:
        """Apply institutional-grade filters"""
        try:
            # Minimum thresholds for institutional consideration
            if signal.strength < self.screening_thresholds['min_confidence']:
                return False
            
            if signal.impact_score < self.screening_thresholds['min_impact_score']:
                return False
            
            if signal.risk_level == 'high' and signal.strength < 0.8:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying institutional filters: {e}")
            return False
    
    def get_screening_summary(self, signals: List[ScreenerSignal]) -> Dict[str, Any]:
        """Generate professional screening summary"""
        try:
            total_signals = len(signals)
            
            if total_signals == 0:
                return {
                    'total_signals': 0,
                    'institutional_summary': 'No signals meet institutional criteria'
                }
            
            # Categorize signals
            bullish_signals = [s for s in signals if s.signal_type == 'bullish']
            bearish_signals = [s for s in signals if s.signal_type == 'bearish']
            high_impact_signals = [s for s in signals if s.impact_score >= 80]
            
            # Calculate averages
            avg_strength = sum(s.strength for s in signals) / total_signals
            avg_impact = sum(s.impact_score for s in signals) / total_signals
            avg_confidence = sum(s.confidence for s in signals) / total_signals
            
            return {
                'total_signals': total_signals,
                'bullish_signals': len(bullish_signals),
                'bearish_signals': len(bearish_signals),
                'high_impact_signals': len(high_impact_signals),
                'avg_strength': avg_strength,
                'avg_impact': avg_impact,
                'avg_confidence': avg_confidence,
                'top_signals': signals[:5],  # Top 5 signals
                'institutional_summary': f"{total_signals} institutional-grade signals identified with {avg_strength:.1%} average strength"
            }
            
        except Exception as e:
            logger.error(f"Error generating screening summary: {e}")
            return {'error': str(e)}