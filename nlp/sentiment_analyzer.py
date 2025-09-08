import logging
import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
from datetime import datetime

from .models import ModelManager
from core.config import NLPConfig
from shared.models import SentimentLabel

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial text"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        
        # Financial sentiment lexicon
        self.positive_terms = [
            'bullish', 'optimistic', 'growth', 'profit', 'gain', 'surge', 'rally',
            'breakthrough', 'success', 'strong', 'beat', 'outperform', 'upgrade',
            'buy', 'positive', 'record', 'milestone', 'approved', 'expansion'
        ]
        
        self.negative_terms = [
            'bearish', 'pessimistic', 'loss', 'decline', 'drop', 'crash', 'sell-off',
            'concern', 'risk', 'weak', 'miss', 'underperform', 'downgrade',
            'sell', 'negative', 'warning', 'caution', 'rejected', 'investigation'
        ]
        
        # Intensity modifiers
        self.intensifiers = [
            'very', 'extremely', 'significantly', 'substantially', 'dramatically',
            'sharply', 'heavily', 'massively', 'tremendously', 'enormous'
        ]
        
        self.diminishers = [
            'slightly', 'somewhat', 'moderately', 'relatively', 'marginally',
            'minor', 'small', 'limited', 'partial', 'gradual'
        ]
        
        logger.info("Sentiment Analyzer initialized")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment of financial text using multiple methods"""
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'score': 0.0,
                    'label': SentimentLabel.NEUTRAL,
                    'confidence': 0.0,
                    'method': 'insufficient_text'
                }
            
            # Try FinBERT first (most accurate for financial text)
            finbert_result = await self._analyze_with_finbert(text)
            if finbert_result:
                return finbert_result
            
            # Fallback to lexicon-based analysis
            lexicon_result = await self._analyze_with_lexicon(text)
            return lexicon_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'score': 0.0,
                'label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    async def _analyze_with_finbert(self, text: str) -> Optional[Dict[str, any]]:
        """Use FinBERT model for sentiment analysis"""
        try:
            finbert_pipeline = await self.model_manager.get_finbert_pipeline()
            if not finbert_pipeline:
                logger.debug("FinBERT model not available")
                return None
            
            # Truncate text to model's max length
            max_length = self.config.max_sequence_length
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get prediction
            result = finbert_pipeline(text)
            
            if result and len(result) > 0:
                prediction = result[0]
                
                # Map FinBERT labels to our sentiment labels
                label_mapping = {
                    'positive': SentimentLabel.POSITIVE,
                    'negative': SentimentLabel.NEGATIVE,
                    'neutral': SentimentLabel.NEUTRAL,
                    'POSITIVE': SentimentLabel.POSITIVE,
                    'NEGATIVE': SentimentLabel.NEGATIVE,
                    'NEUTRAL': SentimentLabel.NEUTRAL
                }
                
                raw_label = prediction['label'].lower()
                sentiment_label = label_mapping.get(raw_label, SentimentLabel.NEUTRAL)
                
                # Convert score to range [-1, 1]
                score = prediction['score']
                if sentiment_label == SentimentLabel.NEGATIVE:
                    score = -score
                elif sentiment_label == SentimentLabel.NEUTRAL:
                    score = 0.0
                
                return {
                    'score': score,
                    'label': sentiment_label,
                    'confidence': prediction['score'],
                    'method': 'finbert',
                    'raw_prediction': prediction
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return None
    
    async def _analyze_with_lexicon(self, text: str) -> Dict[str, any]:
        """Lexicon-based sentiment analysis as fallback"""
        try:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            if not words:
                return {
                    'score': 0.0,
                    'label': SentimentLabel.NEUTRAL,
                    'confidence': 0.0,
                    'method': 'lexicon_no_words'
                }
            
            positive_score = 0
            negative_score = 0
            intensity_multiplier = 1.0
            
            for i, word in enumerate(words):
                # Check for intensity modifiers
                if word in self.intensifiers:
                    intensity_multiplier = 1.5
                    continue
                elif word in self.diminishers:
                    intensity_multiplier = 0.5
                    continue
                
                # Check sentiment terms
                if word in self.positive_terms:
                    positive_score += intensity_multiplier
                elif word in self.negative_terms:
                    negative_score += intensity_multiplier
                
                # Reset multiplier after each sentiment word
                intensity_multiplier = 1.0
            
            # Calculate final score
            total_score = positive_score - negative_score
            max_possible_score = len(words) * 0.1  # Normalize by text length
            
            if max_possible_score > 0:
                normalized_score = total_score / max_possible_score
                # Clamp to [-1, 1]
                normalized_score = max(-1.0, min(1.0, normalized_score))
            else:
                normalized_score = 0.0
            
            # Determine label
            if normalized_score > 0.1:
                sentiment_label = SentimentLabel.POSITIVE
            elif normalized_score < -0.1:
                sentiment_label = SentimentLabel.NEGATIVE
            else:
                sentiment_label = SentimentLabel.NEUTRAL
            
            # Calculate confidence based on the strength of sentiment indicators
            total_sentiment_words = positive_score + negative_score
            confidence = min(1.0, total_sentiment_words / (len(words) * 0.1))
            
            return {
                'score': normalized_score,
                'label': sentiment_label,
                'confidence': confidence,
                'method': 'lexicon',
                'positive_words': positive_score,
                'negative_words': negative_score,
                'total_words': len(words)
            }
            
        except Exception as e:
            logger.error(f"Error in lexicon sentiment analysis: {e}")
            return {
                'score': 0.0,
                'label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'method': 'lexicon_error',
                'error': str(e)
            }
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Analyze sentiment for a batch of texts"""
        try:
            if not texts:
                return []
            
            # For small batches, process sequentially to avoid overwhelming the model
            if len(texts) <= 10:
                results = []
                for text in texts:
                    result = await self.analyze_sentiment(text)
                    results.append(result)
                return results
            
            # For larger batches, use concurrent processing with limited concurrency
            semaphore = asyncio.Semaphore(5)  # Limit concurrent analyses
            
            async def analyze_with_semaphore(text):
                async with semaphore:
                    return await self.analyze_sentiment(text)
            
            tasks = [analyze_with_semaphore(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch sentiment analysis error: {result}")
                    final_results.append({
                        'score': 0.0,
                        'label': SentimentLabel.NEUTRAL,
                        'confidence': 0.0,
                        'method': 'batch_error',
                        'error': str(result)
                    })
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [{
                'score': 0.0,
                'label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'method': 'batch_error',
                'error': str(e)
            } for _ in texts]
    
    def get_sentiment_summary(self, sentiments: List[Dict[str, any]]) -> Dict[str, any]:
        """Get summary statistics for a list of sentiment analyses"""
        try:
            if not sentiments:
                return {
                    'average_score': 0.0,
                    'overall_label': SentimentLabel.NEUTRAL,
                    'confidence': 0.0,
                    'distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
            
            scores = [s.get('score', 0.0) for s in sentiments]
            confidences = [s.get('confidence', 0.0) for s in sentiments]
            labels = [s.get('label', SentimentLabel.NEUTRAL) for s in sentiments]
            
            # Calculate averages
            avg_score = np.mean(scores) if scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Determine overall label
            if avg_score > 0.1:
                overall_label = SentimentLabel.POSITIVE
            elif avg_score < -0.1:
                overall_label = SentimentLabel.NEGATIVE
            else:
                overall_label = SentimentLabel.NEUTRAL
            
            # Calculate distribution
            distribution = {
                'positive': sum(1 for label in labels if label == SentimentLabel.POSITIVE),
                'negative': sum(1 for label in labels if label == SentimentLabel.NEGATIVE),
                'neutral': sum(1 for label in labels if label == SentimentLabel.NEUTRAL)
            }
            
            return {
                'average_score': float(avg_score),
                'overall_label': overall_label,
                'confidence': float(avg_confidence),
                'distribution': distribution,
                'total_analyzed': len(sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment summary: {e}")
            return {
                'average_score': 0.0,
                'overall_label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'error': str(e)
            }
    
    async def cleanup(self):
        """Cleanup sentiment analysis resources"""
        try:
            # Model manager will handle cleanup
            logger.info("Sentiment Analyzer cleanup completed")
        except Exception as e:
            logger.error(f"Error during sentiment analyzer cleanup: {e}")
