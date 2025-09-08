import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import os
import json

# Try to import full NLP components, fallback to simple implementations
try:
    from .models import ModelManager
    from .sentiment_analyzer import SentimentAnalyzer
    from .rule_based_detector import rule_based_detector
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    # Import simple fallback implementations
    from .simple_models import SimpleModelManager as ModelManager
    from .simple_models import SimpleSentimentAnalyzer as SentimentAnalyzer
from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType, NewsItem
from core.config import AppConfig

logger = logging.getLogger(__name__)

class NLPProcessor:
    """Main NLP processor for financial catalyst detection and analysis"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize components based on processing mode
        processing_mode = getattr(config.nlp, 'processing_mode', 'ai')
        
        if processing_mode in ['ai', 'hybrid']:
            self.model_manager = ModelManager(config.nlp)
            self.sentiment_analyzer = SentimentAnalyzer(config.nlp)
        else:
            self.model_manager = None
            self.sentiment_analyzer = None
        
        # Always have rule-based detector available
        self.rule_based_detector = rule_based_detector
        
        self._processing_lock = threading.Lock()
        self._stats = {
            'total_processed': 0,
            'catalysts_detected': 0,
            'avg_processing_time': 0.0,
            'last_processed': None,
            'processing_mode': processing_mode
        }
        
        # Category detection patterns
        self.category_patterns = self._load_category_patterns()
        
        logger.info(f"NLP Processor initialized - Mode: {processing_mode}")
    
    def _load_category_patterns(self) -> Dict[str, List[str]]:
        """Load keyword patterns for category detection"""
        return {
            CatalystType.EARNINGS.value: [
                'earnings', 'revenue', 'profit', 'guidance', 'forecast', 
                'quarterly', 'q1', 'q2', 'q3', 'q4', 'eps', 'beat', 'miss',
                'outlook', 'projections', 'estimates'
            ],
            CatalystType.MA.value: [
                'merger', 'acquisition', 'buyout', 'takeover', 'partnership',
                'joint venture', 'strategic alliance', 'deal', 'agreement',
                'acquire', 'merge', 'bought', 'purchased'
            ],
            CatalystType.REGULATORY.value: [
                'fda', 'approval', 'regulatory', 'clinical trial', 'sec',
                'investigation', 'settlement', 'compliance', 'violation',
                'lawsuit', 'legal', 'court', 'ruling', 'patent'
            ],
            CatalystType.ANALYST.value: [
                'analyst', 'upgrade', 'downgrade', 'rating', 'target price',
                'recommendation', 'buy', 'sell', 'hold', 'neutral',
                'overweight', 'underweight', 'price target'
            ],
            CatalystType.INSIDER.value: [
                'insider', 'ceo', 'cfo', 'executive', 'director', 'officer',
                'insider trading', 'stock purchase', 'stock sale', 'resignation',
                'appointment', 'management change'
            ]
        }
    
    async def process_news_item(self, news_item: NewsItem, stock_info: Optional[Dict] = None) -> Optional[Catalyst]:
        """Process a single news item and extract catalyst information"""
        try:
            start_time = datetime.now()
            
            # Combine title and content for analysis
            text_content = f"{news_item.title or ''}\n\n{news_item.content or ''}".strip()
            
            if not text_content or len(text_content) < 50:
                logger.debug("Skipping news item: insufficient content")
                return None
            
            # Check if content is relevant to financial markets
            if not await self._is_financial_relevant(text_content):
                logger.debug("Skipping news item: not financially relevant")
                return None
            
            # Detect catalyst category
            category = await self._detect_category(text_content)
            
            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(text_content)
            
            # Calculate impact score
            impact_score = await self._calculate_impact_score(
                text_content, category, sentiment_result, stock_info
            )
            
            # Determine if this qualifies as a catalyst
            if impact_score < self.config.alerts.min_impact_score:
                logger.debug(f"Skipping news item: impact score {impact_score} below threshold")
                return None
            
            # Extract ticker from news item or stock info
            ticker = stock_info.get('ticker') if stock_info else self._extract_ticker(text_content)
            if not ticker:
                logger.debug("Skipping news item: no ticker found")
                return None
            
            # Create catalyst object
            catalyst = Catalyst(
                ticker=ticker.upper(),
                catalyst=self._generate_catalyst_summary(text_content),
                category=category,
                sentiment_label=sentiment_result['label'],
                sector=stock_info.get('sector', 'Unknown') if stock_info else 'Unknown',
                sentiment_score=sentiment_result['score'],
                impact=impact_score,
                source=SourceType(news_item.source.value),
                confidence=sentiment_result['confidence'],
                published_date=news_item.published_date,
                url=news_item.url,
                metadata={
                    'author': news_item.author,
                    'source_name': news_item.source_name,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'content_length': len(text_content)
                }
            )
            
            # Update statistics
            with self._processing_lock:
                self._stats['total_processed'] += 1
                self._stats['catalysts_detected'] += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                self._stats['avg_processing_time'] = (
                    self._stats['avg_processing_time'] * 0.9 + processing_time * 0.1
                )
                self._stats['last_processed'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Catalyst detected: {ticker} - {category.value} (Impact: {impact_score})")
            return catalyst
            
        except Exception as e:
            logger.error(f"Error processing news item: {e}")
            with self._processing_lock:
                self._stats['total_processed'] += 1
            return None
    
    async def process_batch(self, news_items: List[NewsItem], stock_info_map: Optional[Dict[str, Dict]] = None) -> List[Catalyst]:
        """Process a batch of news items concurrently"""
        try:
            if not news_items:
                return []
            
            logger.info(f"Processing batch of {len(news_items)} news items")
            
            # Create tasks for concurrent processing
            tasks = []
            for news_item in news_items:
                # Get stock info for this item if available
                stock_info = None
                if stock_info_map:
                    ticker = self._extract_ticker(news_item.title or news_item.content or '')
                    if ticker:
                        stock_info = stock_info_map.get(ticker.upper())
                
                task = self.process_news_item(news_item, stock_info)
                tasks.append(task)
            
            # Process all items concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            catalysts = []
            for result in results:
                if isinstance(result, Catalyst):
                    catalysts.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
            
            logger.info(f"Batch processing completed: {len(catalysts)} catalysts detected from {len(news_items)} items")
            return catalysts
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    async def _is_financial_relevant(self, text: str) -> bool:
        """Determine if text content is financially relevant"""
        try:
            financial_keywords = [
                'stock', 'shares', 'market', 'trading', 'price', 'revenue', 'earnings',
                'profit', 'loss', 'investment', 'investor', 'financial', 'quarter',
                'analyst', 'forecast', 'guidance', 'performance', 'growth', 'decline'
            ]
            
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
            
            # Consider relevant if it contains multiple financial keywords
            return keyword_count >= 2
            
        except Exception as e:
            logger.error(f"Error checking financial relevance: {e}")
            return True  # Default to relevant to avoid false negatives
    
    async def _detect_category(self, text: str) -> CatalystType:
        """Detect catalyst category from text content"""
        try:
            text_lower = text.lower()
            category_scores = {}
            
            # Score each category based on keyword matches
            for category, keywords in self.category_patterns.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        # Weight longer keywords higher
                        score += len(keyword.split())
                category_scores[category] = score
            
            # Return category with highest score, or GENERAL if no strong match
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:
                    return CatalystType(best_category[0])
            
            return CatalystType.GENERAL
            
        except Exception as e:
            logger.error(f"Error detecting category: {e}")
            return CatalystType.GENERAL
    
    async def _calculate_impact_score(self, text: str, category: CatalystType, 
                                    sentiment_result: Dict, stock_info: Optional[Dict] = None) -> int:
        """Calculate impact score for a potential catalyst"""
        try:
            base_score = 50  # Base score
            
            # Category impact weights
            category_weights = {
                CatalystType.EARNINGS: 25,
                CatalystType.MA: 30,
                CatalystType.REGULATORY: 20,
                CatalystType.ANALYST: 15,
                CatalystType.INSIDER: 10,
                CatalystType.GENERAL: 5
            }
            
            # Add category weight
            base_score += category_weights.get(category, 0)
            
            # Add sentiment impact
            sentiment_impact = abs(sentiment_result['score']) * 20
            base_score += sentiment_impact
            
            # Add confidence impact
            confidence_impact = sentiment_result['confidence'] * 15
            base_score += confidence_impact
            
            # Boost for high-impact keywords
            high_impact_keywords = [
                'breakthrough', 'record', 'milestone', 'approved', 'rejected',
                'acquisition', 'merger', 'lawsuit', 'investigation', 'bankruptcy'
            ]
            
            text_lower = text.lower()
            for keyword in high_impact_keywords:
                if keyword in text_lower:
                    base_score += 10
            
            # Market cap adjustment (if available)
            if stock_info and stock_info.get('market_cap'):
                market_cap = stock_info['market_cap']
                if market_cap > 100_000_000_000:  # Large cap
                    base_score += 5
                elif market_cap < 2_000_000_000:  # Small cap
                    base_score += 10
            
            # Ensure score is within bounds
            return max(0, min(100, int(base_score)))
            
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 50  # Default middle score
    
    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract stock ticker from text content"""
        try:
            import re
            
            # Look for ticker patterns
            patterns = [
                r'\$([A-Z]{1,5})\b',  # $TICKER
                r'\b([A-Z]{2,5})\s+stock\b',  # TICKER stock
                r'\b([A-Z]{2,5})\s+shares\b',  # TICKER shares
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text.upper())
                for match in matches:
                    if len(match) >= 2 and len(match) <= 5:
                        return match
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting ticker: {e}")
            return None
    
    def _generate_catalyst_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a concise summary of the catalyst"""
        try:
            # Take first sentence or first 200 characters
            sentences = text.split('.')
            if sentences and len(sentences[0]) <= max_length:
                return sentences[0].strip() + '.'
            
            # Fallback to truncation
            if len(text) <= max_length:
                return text.strip()
            
            truncated = text[:max_length].rsplit(' ', 1)[0]
            return truncated.strip() + '...'
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + '...' if len(text) > 200 else text
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of NLP models"""
        return self.model_manager.get_status()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self._processing_lock:
            return dict(self._stats)
    
    async def cleanup(self):
        """Cleanup NLP resources"""
        try:
            await self.model_manager.cleanup()
            await self.sentiment_analyzer.cleanup()
            logger.info("NLP Processor cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during NLP cleanup: {e}")
