import logging
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class SimpleModelManager:
    """Simple fallback model manager when ML dependencies are not available"""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'
        self._models = {}
        logger.info("Simple Model Manager initialized (fallback mode)")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of models - fallback implementation"""
        return {
            'loaded': True,
            'models': {
                'finbert': {'loaded': False, 'device': 'cpu', 'error': 'Using simple rule-based analysis'},
                'bart': {'loaded': False, 'device': 'cpu', 'error': 'Using simple summarization'},
                'spacy': {'loaded': False, 'device': 'cpu', 'error': 'Using simple NLP processing'}
            },
            'loaded_count': 0,
            'total_models': 3,
            'device': 'cpu'
        }
    
    async def preload_all_models(self) -> bool:
        """Preload models - fallback always returns True"""
        logger.info("Using simple fallback implementations")
        return True

class SimpleSentimentAnalyzer:
    """Simple rule-based sentiment analyzer"""
    
    def __init__(self, config):
        self.config = config
        self.model_manager = SimpleModelManager(config)
        
        # Basic sentiment keywords
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
        
        logger.info("Simple Sentiment Analyzer initialized")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis"""
        if not text:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0, 'method': 'simple'}
        
        text_lower = text.lower()
        
        # Count positive and negative terms
        pos_count = sum(1 for term in self.positive_terms if term in text_lower)
        neg_count = sum(1 for term in self.negative_terms if term in text_lower)
        
        # Calculate simple score
        if pos_count > neg_count:
            score = min(0.8, 0.3 + (pos_count - neg_count) * 0.1)
            label = 'positive'
        elif neg_count > pos_count:
            score = max(-0.8, -0.3 - (neg_count - pos_count) * 0.1)
            label = 'negative'
        else:
            score = 0.0
            label = 'neutral'
        
        confidence = min(0.9, abs(score) + 0.1)
        
        return {
            'score': score,
            'label': label,
            'confidence': confidence,
            'method': 'simple_rule_based'
        }

class SimpleNLPProcessor:
    """Simple NLP processor with basic functionality"""
    
    def __init__(self, config):
        self.config = config
        self.model_manager = SimpleModelManager(config.nlp)
        self.sentiment_analyzer = SimpleSentimentAnalyzer(config.nlp)
        
        self._stats = {
            'total_processed': 0,
            'catalysts_detected': 0,
            'avg_processing_time': 0.0,
            'last_processed': None
        }
        
        logger.info("Simple NLP Processor initialized")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status"""
        return self.model_manager.get_model_status()
    
    async def process_news_batch(self, news_items) -> list:
        """Process news items - simplified implementation"""
        results = []
        for item in news_items:
            # Basic catalyst detection based on keywords
            catalyst_keywords = ['earnings', 'merger', 'acquisition', 'fda', 'approval', 'partnership']
            text = (item.get('title', '') + ' ' + item.get('description', '')).lower()
            
            # Check if it might be a catalyst
            is_catalyst = any(keyword in text for keyword in catalyst_keywords)
            
            if is_catalyst:
                sentiment = await self.sentiment_analyzer.analyze_sentiment(text)
                
                result = {
                    'ticker': item.get('ticker', 'UNKNOWN'),
                    'catalyst': item.get('title', 'News Item'),
                    'category': 'General News',
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'impact': min(100, max(10, int(abs(sentiment['score']) * 100))),
                    'confidence': sentiment['confidence'],
                    'source': item.get('source', 'RSS'),
                    'url': item.get('url', ''),
                    'published_date': item.get('published_date'),
                    'extra_data': {'method': 'simple_processing'}
                }
                results.append(result)
        
        self._stats['total_processed'] += len(news_items)
        self._stats['catalysts_detected'] += len(results)
        
        return results