import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import os
import json
import re
import numpy as np
from textblob import TextBlob

# Import existing simple models as fallback
try:
    from .simple_models import SimpleModelManager, SimpleSentimentAnalyzer
except ImportError:
    SimpleModelManager = None
    SimpleSentimentAnalyzer = None

from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType, NewsItem
from core.config import AppConfig

logger = logging.getLogger(__name__)

class AdvancedNLPProcessor:
    """Advanced NLP processor with financial-specific models and ensemble methods"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._processing_lock = threading.Lock()
        self._stats = {
            'total_processed': 0,
            'catalysts_detected': 0,
            'avg_processing_time': 0.0,
            'last_processed': None,
            'sentiment_accuracy': 0.0,
            'category_accuracy': 0.0
        }
        
        # Financial keyword dictionaries for enhanced analysis
        self.financial_keywords = self._load_financial_keywords()
        self.catalyst_patterns = self._load_catalyst_patterns()
        self.sentiment_modifiers = self._load_sentiment_modifiers()
        
        # Trading signal weights for different catalyst types
        self.trading_weights = {
            CatalystType.EARNINGS.value: {
                'day_trading': 0.9,
                'momentum': 0.85,
                'long_term': 0.7
            },
            CatalystType.MA.value: {
                'day_trading': 1.0,
                'momentum': 0.95,
                'long_term': 0.9
            },
            CatalystType.REGULATORY.value: {
                'day_trading': 0.8,
                'momentum': 0.9,
                'long_term': 0.95
            },
            CatalystType.ANALYST.value: {
                'day_trading': 0.7,
                'momentum': 0.8,
                'long_term': 0.6
            }
        }
        
        logger.info("Advanced NLP Processor initialized")
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive financial keyword dictionaries"""
        return {
            'bullish_indicators': [
                'beat expectations', 'exceeded estimates', 'record revenue', 'record profit',
                'strong growth', 'positive outlook', 'raised guidance', 'upgraded',
                'breakthrough', 'milestone', 'approved', 'partnership', 'acquisition',
                'expansion', 'innovation', 'market leader', 'competitive advantage'
            ],
            'bearish_indicators': [
                'missed estimates', 'below expectations', 'declining revenue', 'loss',
                'weak performance', 'lowered guidance', 'downgraded', 'investigation',
                'lawsuit', 'regulatory issues', 'bankruptcy', 'debt', 'restructuring',
                'layoffs', 'management issues', 'competitive pressure'
            ],
            'volatility_indicators': [
                'merger', 'acquisition', 'takeover', 'fda approval', 'clinical trial',
                'earnings surprise', 'guidance revision', 'insider trading',
                'unusual options activity', 'short squeeze', 'analyst revision'
            ],
            'momentum_indicators': [
                'breakout', 'rally', 'surge', 'momentum', 'trend reversal',
                'volume spike', 'technical analysis', 'support level', 'resistance level',
                'moving average', 'chart pattern', 'institutional buying'
            ]
        }
    
    def _load_catalyst_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load advanced catalyst detection patterns"""
        return {
            CatalystType.EARNINGS.value: {
                'keywords': ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'guidance'],
                'patterns': [r'Q[1-4]\s+\d{4}', r'earnings\s+beat', r'revenue\s+growth'],
                'impact_multiplier': 1.2,
                'trading_relevance': {'day': 0.9, 'swing': 0.8, 'long': 0.7}
            },
            CatalystType.MA.value: {
                'keywords': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
                'patterns': [r'\$[0-9]+\s*billion\s*deal', r'acquire.*for\s*\$'],
                'impact_multiplier': 1.4,
                'trading_relevance': {'day': 1.0, 'swing': 0.95, 'long': 0.9}
            },
            CatalystType.REGULATORY.value: {
                'keywords': ['fda', 'approval', 'clinical', 'sec', 'regulatory'],
                'patterns': [r'FDA\s+approval', r'clinical\s+trial\s+results'],
                'impact_multiplier': 1.3,
                'trading_relevance': {'day': 0.8, 'swing': 0.9, 'long': 0.95}
            }
        }
    
    def _load_sentiment_modifiers(self) -> Dict[str, float]:
        """Load sentiment modification weights for financial context"""
        return {
            'strongly positive': 0.9,
            'positive': 0.6,
            'slightly positive': 0.3,
            'neutral': 0.0,
            'slightly negative': -0.3,
            'negative': -0.6,
            'strongly negative': -0.9,
            # Financial specific modifiers
            'beat expectations': 0.8,
            'exceeded guidance': 0.7,
            'missed estimates': -0.8,
            'below guidance': -0.7,
            'record high': 0.9,
            'all-time high': 0.9,
            'bankruptcy': -1.0,
            'investigation': -0.8
        }
    
    async def analyze_financial_sentiment(self, text: str, ticker: str = None) -> Dict[str, Any]:
        """Advanced financial sentiment analysis with ensemble approach"""
        try:
            # Clean and prepare text
            cleaned_text = self._clean_financial_text(text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 10:
                return {'score': 0.0, 'label': SentimentLabel.NEUTRAL, 'confidence': 0.0}
            
            # Multiple sentiment analysis approaches
            sentiments = []
            confidences = []
            
            # 1. TextBlob baseline (always available)
            try:
                blob = TextBlob(cleaned_text)
                textblob_score = blob.sentiment.polarity
                sentiments.append(textblob_score)
                confidences.append(0.6)  # Base confidence for TextBlob
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
            
            # 2. Financial keyword analysis
            keyword_sentiment, keyword_confidence = self._analyze_financial_keywords(cleaned_text)
            if keyword_confidence > 0:
                sentiments.append(keyword_sentiment)
                confidences.append(keyword_confidence)
            
            # 3. Pattern-based analysis
            pattern_sentiment, pattern_confidence = self._analyze_sentiment_patterns(cleaned_text)
            if pattern_confidence > 0:
                sentiments.append(pattern_sentiment)
                confidences.append(pattern_confidence)
            
            # 4. Context-aware analysis (if ticker provided)
            if ticker:
                context_sentiment, context_confidence = self._analyze_ticker_context(cleaned_text, ticker)
                if context_confidence > 0:
                    sentiments.append(context_sentiment)
                    confidences.append(context_confidence)
            
            # Ensemble combination
            if not sentiments:
                return {'score': 0.0, 'label': SentimentLabel.NEUTRAL, 'confidence': 0.0}
            
            # Weighted average based on confidence
            total_weight = sum(confidences)
            if total_weight == 0:
                weighted_score = sum(sentiments) / len(sentiments)
                avg_confidence = 0.5
            else:
                weighted_score = sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
                avg_confidence = total_weight / len(confidences)
            
            # Determine label
            if weighted_score > 0.2:
                label = SentimentLabel.POSITIVE
            elif weighted_score < -0.2:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
            
            return {
                'score': float(np.clip(weighted_score, -1.0, 1.0)),
                'label': label,
                'confidence': float(np.clip(avg_confidence, 0.0, 1.0)),
                'components': {
                    'sentiment_count': len(sentiments),
                    'raw_scores': sentiments,
                    'confidences': confidences
                }
            }
            
        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {e}")
            return {'score': 0.0, 'label': SentimentLabel.NEUTRAL, 'confidence': 0.0}
    
    def _clean_financial_text(self, text: str) -> str:
        """Clean and normalize financial text for analysis"""
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Normalize financial abbreviations
            text = re.sub(r'\b(Q[1-4])\b', r'quarter \1', text)
            text = re.sub(r'\b(FY)\s*(\d{2,4})\b', r'fiscal year \2', text)
            text = re.sub(r'\b(YoY|y/y)\b', 'year over year', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(QoQ|q/q)\b', 'quarter over quarter', text, flags=re.IGNORECASE)
            
            # Normalize currency and percentages
            text = re.sub(r'\$([0-9,.]+)([BMK])?', r'\1 \2 dollars', text)
            text = re.sub(r'([0-9,.]+)%', r'\1 percent', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning financial text: {e}")
            return text
    
    def _analyze_financial_keywords(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment based on financial keywords"""
        try:
            text_lower = text.lower()
            score = 0.0
            total_matches = 0
            
            # Analyze bullish indicators
            for keyword in self.financial_keywords['bullish_indicators']:
                if keyword in text_lower:
                    score += 0.7
                    total_matches += 1
            
            # Analyze bearish indicators  
            for keyword in self.financial_keywords['bearish_indicators']:
                if keyword in text_lower:
                    score -= 0.7
                    total_matches += 1
            
            # Calculate confidence based on matches
            confidence = min(1.0, total_matches * 0.2)
            
            if total_matches > 0:
                score = score / total_matches  # Normalize by number of matches
            
            return float(np.clip(score, -1.0, 1.0)), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            return 0.0, 0.0
    
    def _analyze_sentiment_patterns(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using regex patterns"""
        try:
            patterns = {
                'strong_positive': [
                    r'beat.*expectations?.*by.*[0-9]+%',
                    r'record.*(?:revenue|profit|earnings)',
                    r'exceeded.*estimates?.*by.*[0-9]+%'
                ],
                'positive': [
                    r'better.*than.*expected',
                    r'strong.*(?:growth|performance)',
                    r'raised.*guidance'
                ],
                'negative': [
                    r'missed.*estimates?',
                    r'below.*expectations?',
                    r'lowered.*guidance'
                ],
                'strong_negative': [
                    r'bankruptcy.*filed',
                    r'sec.*investigation',
                    r'massive.*loss'
                ]
            }
            
            scores = {'strong_positive': 0.8, 'positive': 0.4, 'negative': -0.4, 'strong_negative': -0.8}
            total_score = 0.0
            total_matches = 0
            
            for sentiment_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, text, re.IGNORECASE):
                        total_score += scores[sentiment_type]
                        total_matches += 1
            
            confidence = min(1.0, total_matches * 0.3)
            
            if total_matches > 0:
                avg_score = total_score / total_matches
            else:
                avg_score = 0.0
            
            return float(np.clip(avg_score, -1.0, 1.0)), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return 0.0, 0.0
    
    def _analyze_ticker_context(self, text: str, ticker: str) -> Tuple[float, float]:
        """Analyze sentiment in the context of a specific ticker"""
        try:
            # Look for ticker-specific mentions
            ticker_patterns = [
                f'\${ticker}\b',
                f'\b{ticker}\s+(?:stock|shares)\b',
                f'\b{ticker}\b.*(?:up|down|gains?|loss)'
            ]
            
            ticker_mentions = 0
            context_score = 0.0
            
            for pattern in ticker_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                ticker_mentions += len(matches)
            
            if ticker_mentions == 0:
                return 0.0, 0.0
            
            # Analyze context around ticker mentions
            ticker_context = re.findall(
                f'(.{{0,50}}\b{ticker}\b.{{0,50}})', 
                text, 
                re.IGNORECASE | re.DOTALL
            )
            
            for context in ticker_context:
                # Simple positive/negative word analysis in context
                positive_words = ['up', 'gain', 'rise', 'bull', 'strong', 'buy', 'upgrade']
                negative_words = ['down', 'drop', 'fall', 'bear', 'weak', 'sell', 'downgrade']
                
                context_lower = context.lower()
                pos_count = sum(1 for word in positive_words if word in context_lower)
                neg_count = sum(1 for word in negative_words if word in context_lower)
                
                if pos_count > neg_count:
                    context_score += 0.5
                elif neg_count > pos_count:
                    context_score -= 0.5
            
            confidence = min(1.0, ticker_mentions * 0.4)
            
            if len(ticker_context) > 0:
                avg_score = context_score / len(ticker_context)
            else:
                avg_score = 0.0
            
            return float(np.clip(avg_score, -1.0, 1.0)), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in ticker context analysis: {e}")
            return 0.0, 0.0
    
    async def categorize_catalyst_advanced(self, text: str, ticker: str = None) -> Tuple[CatalystType, float]:
        """Advanced catalyst categorization with confidence scoring"""
        try:
            text_lower = text.lower()
            category_scores = {}
            
            # Analyze each catalyst pattern
            for category, pattern_info in self.catalyst_patterns.items():
                score = 0
                
                # Keyword matching
                for keyword in pattern_info['keywords']:
                    if keyword in text_lower:
                        score += 2
                
                # Regex pattern matching
                for pattern in pattern_info.get('patterns', []):
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 3
                
                # Apply impact multiplier
                score *= pattern_info.get('impact_multiplier', 1.0)
                
                category_scores[category] = score
            
            # Find best match
            if category_scores:
                best_category, best_score = max(category_scores.items(), key=lambda x: x[1])
                
                # Calculate confidence based on score difference
                sorted_scores = sorted(category_scores.values(), reverse=True)
                if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                    confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                else:
                    confidence = min(1.0, best_score / 5.0)
                
                if best_score > 0:
                    return CatalystType(best_category), float(confidence)
            
            return CatalystType.GENERAL, 0.3
            
        except Exception as e:
            logger.error(f"Error in advanced categorization: {e}")
            return CatalystType.GENERAL, 0.0
    
    async def calculate_trading_impact_score(self, text: str, category: CatalystType, 
                                          sentiment_result: Dict, trading_type: str = 'day_trading') -> int:
        """Calculate impact score specifically for trading opportunities"""
        try:
            base_score = 50
            
            # Get category-specific trading weights
            category_key = category.value
            if category_key in self.trading_weights:
                trading_weight = self.trading_weights[category_key].get(trading_type.replace('_', ''), 1.0)
                base_score *= trading_weight
            
            # Sentiment impact (stronger for day trading)
            sentiment_multiplier = 1.5 if trading_type == 'day_trading' else 1.2
            sentiment_impact = abs(sentiment_result['score']) * 25 * sentiment_multiplier
            base_score += sentiment_impact
            
            # Confidence boost
            confidence_boost = sentiment_result['confidence'] * 15
            base_score += confidence_boost
            
            # Trading-specific keyword analysis
            text_lower = text.lower()
            
            # High-volatility indicators (great for day trading)
            volatility_keywords = self.financial_keywords.get('volatility_indicators', [])
            volatility_score = sum(10 for keyword in volatility_keywords if keyword in text_lower)
            
            if trading_type == 'day_trading':
                base_score += volatility_score * 1.5
            else:
                base_score += volatility_score
            
            # Momentum indicators (great for swing trading)
            momentum_keywords = self.financial_keywords.get('momentum_indicators', [])
            momentum_score = sum(8 for keyword in momentum_keywords if keyword in text_lower)
            
            if 'momentum' in trading_type:
                base_score += momentum_score * 1.3
            else:
                base_score += momentum_score
            
            # Time sensitivity analysis
            time_sensitive_patterns = [
                r'\btoday\b', r'\bthis morning\b', r'\bjust announced\b',
                r'\bbreaking\b', r'\burgent\b', r'\balert\b'
            ]
            
            time_sensitive = any(re.search(pattern, text, re.IGNORECASE) 
                               for pattern in time_sensitive_patterns)
            
            if time_sensitive and trading_type == 'day_trading':
                base_score += 20
            elif time_sensitive:
                base_score += 10
            
            # Ensure score bounds
            final_score = max(0, min(100, int(base_score)))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating trading impact score: {e}")
            return 50
    
    def get_trading_signal_strength(self, catalyst: Catalyst, trading_type: str = 'day_trading') -> Dict[str, Any]:
        """Determine trading signal strength based on catalyst analysis"""
        try:
            # Base signal strength from impact score
            base_strength = catalyst.impact / 100.0
            
            # Adjust for trading type
            category_key = catalyst.category.value
            if category_key in self.trading_weights:
                type_adjustment = self.trading_weights[category_key].get(
                    trading_type.replace('_', ''), 1.0
                )
                base_strength *= type_adjustment
            
            # Sentiment direction
            sentiment_direction = 1 if catalyst.sentiment_score > 0 else -1
            
            # Signal classification
            if base_strength >= 0.8:
                signal_class = 'STRONG BUY' if sentiment_direction > 0 else 'STRONG SELL'
            elif base_strength >= 0.6:
                signal_class = 'BUY' if sentiment_direction > 0 else 'SELL'
            elif base_strength >= 0.4:
                signal_class = 'WEAK BUY' if sentiment_direction > 0 else 'WEAK SELL'
            else:
                signal_class = 'HOLD'
            
            return {
                'strength': float(base_strength),
                'direction': sentiment_direction,
                'signal': signal_class,
                'confidence': float(catalyst.confidence),
                'trading_type': trading_type,
                'risk_level': 'HIGH' if base_strength > 0.7 else 'MEDIUM' if base_strength > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading signal strength: {e}")
            return {
                'strength': 0.5,
                'direction': 0,
                'signal': 'HOLD',
                'confidence': 0.0,
                'trading_type': trading_type,
                'risk_level': 'MEDIUM'
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self._processing_lock:
            return dict(self._stats)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Advanced NLP Processor cleaned up")
