import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
import time

from data_fetchers import get_all_fetchers
from nlp.processor import NLPProcessor
from shared.models import Catalyst, NewsItem, SourceType
from core.config import AppConfig
from core.database import DatabaseManager
from core.cache import CacheManager

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing of financial data with efficient concurrent operations"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize components (would be injected in full system)
        self.db_manager = DatabaseManager(config)
        self.cache_manager = CacheManager(config)
        self.nlp_processor = NLPProcessor(config)
        
        # Processing statistics
        self._stats = {
            'batches_processed': 0,
            'total_items_processed': 0,
            'catalysts_generated': 0,
            'processing_time_total': 0.0,
            'last_batch_time': None,
            'avg_processing_time': 0.0
        }
        
        # Enhanced concurrent processing limits
        self.max_concurrent_fetchers = config.fetchers.max_concurrent  # Now 25
        self.max_concurrent_nlp = config.fetchers.concurrent_nlp_workers  # Now 10
        self.worker_pool_size = config.fetchers.worker_pool_size  # 15
        
        # Cross-validation settings
        self.cross_validation_enabled = config.fetchers.cross_validation_enabled
        self.min_sources_for_validation = getattr(config.nlp, 'min_sources_for_validation', 2)
        
        logger.info("Batch Processor initialized")
    
    async def process_ticker_batch(self, tickers: List[str], sources: List[str] = None, 
                                 limit_per_source: int = 20) -> Dict[str, List[Catalyst]]:
        """Process a batch of tickers from multiple sources"""
        start_time = time.time()
        
        try:
            if sources is None:
                sources = ['newsapi', 'twitter', 'reddit', 'rss']
            
            logger.info(f"Starting batch processing: {len(tickers)} tickers, {len(sources)} sources")
            
            # Get configured fetchers
            all_fetchers = get_all_fetchers(self.config)
            active_fetchers = {
                name: fetcher for name, fetcher in all_fetchers.items()
                if name in sources and fetcher.is_configured()
            }
            
            if not active_fetchers:
                logger.warning("No active fetchers available for batch processing")
                return {}
            
            logger.info(f"Using {len(active_fetchers)} active fetchers: {list(active_fetchers.keys())}")
            
            # Process all tickers concurrently
            semaphore = asyncio.Semaphore(self.max_concurrent_fetchers)
            
            async def process_single_ticker(ticker: str) -> tuple[str, List[Catalyst]]:
                async with semaphore:
                    return ticker, await self._process_ticker_all_sources(
                        ticker, active_fetchers, limit_per_source
                    )
            
            # Create tasks for all tickers
            tasks = [process_single_ticker(ticker) for ticker in tickers]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            final_results = {}
            total_catalysts = 0
            
            for result in results:
                if isinstance(result, tuple):
                    ticker, catalysts = result
                    final_results[ticker] = catalysts
                    total_catalysts += len(catalysts)
                elif isinstance(result, Exception):
                    logger.error(f"Ticker processing failed: {result}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(tickers), total_catalysts, processing_time)
            
            logger.info(f"Batch processing completed: {total_catalysts} catalysts from {len(tickers)} tickers in {processing_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {}
    
    async def _process_ticker_all_sources(self, ticker: str, fetchers: Dict[str, Any], 
                                        limit_per_source: int) -> List[Catalyst]:
        """Process a single ticker across all available sources"""
        try:
            # Check cache first
            cache_key = f"batch_ticker_{ticker}_{limit_per_source}"
            cached_result = self.cache_manager.get(cache_key, namespace="batch_processing")
            if cached_result:
                logger.debug(f"Cache hit for ticker {ticker}")
                return cached_result
            
            # Fetch data from all sources concurrently
            fetch_tasks = []
            for source_name, fetcher in fetchers.items():
                task = self._safe_fetch(fetcher, ticker, limit_per_source, source_name)
                fetch_tasks.append(task)
            
            # Wait for all fetches to complete
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Combine all news items
            all_news_items = []
            for result in fetch_results:
                if isinstance(result, list):
                    all_news_items.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Fetch failed for {ticker}: {result}")
            
            logger.debug(f"Fetched {len(all_news_items)} items for {ticker}")
            
            if not all_news_items:
                return []
            
            # Convert to NewsItem objects and process with NLP
            news_items = self._convert_to_news_items(all_news_items, ticker)
            catalysts = await self._process_news_items_batch(news_items, ticker)
            
            # Cache the results
            self.cache_manager.set(cache_key, catalysts, ttl=300, namespace="batch_processing")
            
            # Save catalysts to database
            await self._save_catalysts_batch(catalysts)
            
            return catalysts
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            return []
    
    async def _safe_fetch(self, fetcher, ticker: str, limit: int, source_name: str) -> List[Dict]:
        """Safely fetch data from a source with error handling"""
        try:
            start_time = time.time()
            data = await fetcher.fetch_all(ticker, limit)
            fetch_time = time.time() - start_time
            
            logger.debug(f"Fetched {len(data)} items from {source_name} for {ticker} in {fetch_time:.2f}s")
            return data
            
        except Exception as e:
            logger.warning(f"Fetch failed for {source_name}/{ticker}: {e}")
            return []
    
    def _convert_to_news_items(self, raw_items: List[Dict], ticker: str) -> List[NewsItem]:
        """Convert raw data items to NewsItem objects"""
        news_items = []
        
        for item in raw_items:
            try:
                # Determine source type
                source_type = SourceType.UNKNOWN
                if 'source' in item:
                    source_value = item['source']
                    try:
                        source_type = SourceType(source_value)
                    except ValueError:
                        # Try to map common source names
                        source_mapping = {
                            'newsapi': SourceType.NEWSAPI,
                            'twitter': SourceType.TWITTER,
                            'reddit': SourceType.REDDIT,
                            'rss': SourceType.RSS,
                            'regulatory': SourceType.REGULATORY,
                            'financial': SourceType.FINANCIAL
                        }
                        source_type = source_mapping.get(source_value.lower(), SourceType.UNKNOWN)
                
                # Create NewsItem
                news_item = NewsItem(
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    source=source_type,
                    published_date=self._parse_date(item.get('published_date')),
                    url=item.get('url', ''),
                    author=item.get('author', ''),
                    source_name=item.get('source_name', source_type.value)
                )
                
                news_items.append(news_item)
                
            except Exception as e:
                logger.warning(f"Failed to convert item to NewsItem: {e}")
                continue
        
        return news_items
    
    def _parse_date(self, date_value) -> Optional[datetime]:
        """Parse various date formats to datetime"""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            try:
                # Try ISO format
                if 'T' in date_value:
                    return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                else:
                    return datetime.strptime(date_value, '%Y-%m-%d')
            except ValueError:
                logger.debug(f"Could not parse date: {date_value}")
        
        return None
    
    async def _process_news_items_batch(self, news_items: List[NewsItem], ticker: str) -> List[Catalyst]:
        """Process news items through NLP to generate catalysts"""
        try:
            if not news_items:
                return []
            
            # Process in smaller batches to avoid overwhelming the NLP models
            batch_size = self.config.nlp.batch_size
            catalysts = []
            
            for i in range(0, len(news_items), batch_size):
                batch = news_items[i:i + batch_size]
                
                # Process batch with limited concurrency
                semaphore = asyncio.Semaphore(self.max_concurrent_nlp)
                
                async def process_single_item(news_item: NewsItem) -> Optional[Catalyst]:
                    async with semaphore:
                        return await self.nlp_processor.process_news_item(
                            news_item, 
                            stock_info={'ticker': ticker}
                        )
                
                # Create tasks for this batch
                tasks = [process_single_item(item) for item in batch]
                
                # Process batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid catalysts
                for result in batch_results:
                    if isinstance(result, Catalyst):
                        catalysts.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"NLP processing failed: {result}")
            
            # Apply cross-validation if enabled and multiple sources
            if self.cross_validation_enabled and len(catalysts) >= self.min_sources_for_validation:
                catalysts = await self._apply_cross_validation(catalysts, ticker)
            
            logger.debug(f"Generated {len(catalysts)} catalysts from {len(news_items)} news items for {ticker}")
            return catalysts
            
        except Exception as e:
            logger.error(f"Error processing news items batch: {e}")
            return []
    
    async def _save_catalysts_batch(self, catalysts: List[Catalyst]):
        """Save catalysts to database in batch"""
        try:
            if not catalysts:
                return
            
            saved_count = 0
            for catalyst in catalysts:
                try:
                    catalyst_id = self.db_manager.save_catalyst(catalyst)
                    if catalyst_id:
                        saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save catalyst: {e}")
            
            logger.debug(f"Saved {saved_count}/{len(catalysts)} catalysts to database")
            
        except Exception as e:
            logger.error(f"Error saving catalysts batch: {e}")
    
    async def _apply_cross_validation(self, catalysts: List[Catalyst], ticker: str) -> List[Catalyst]:
        """Apply cross-validation to improve catalyst accuracy"""
        try:
            # Group catalysts by similar content/topic
            validated_catalysts = []
            processed_indices = set()
            
            for i, catalyst in enumerate(catalysts):
                if i in processed_indices:
                    continue
                
                # Find similar catalysts (same category, similar keywords)
                similar_catalysts = [catalyst]
                similar_indices = {i}
                
                for j, other_catalyst in enumerate(catalysts[i+1:], start=i+1):
                    if j in processed_indices:
                        continue
                    
                    if self._are_catalysts_similar(catalyst, other_catalyst):
                        similar_catalysts.append(other_catalyst)
                        similar_indices.add(j)
                
                # If we have multiple similar catalysts from different sources, validate
                if len(similar_catalysts) >= 2:
                    validated_catalyst = self._validate_catalyst_group(similar_catalysts)
                    if validated_catalyst:
                        validated_catalysts.append(validated_catalyst)
                else:
                    # Single source catalyst - keep if confidence is high enough
                    if catalyst.confidence >= 0.8:
                        validated_catalysts.append(catalyst)
                
                processed_indices.update(similar_indices)
            
            logger.debug(f"Cross-validation: {len(catalysts)} -> {len(validated_catalysts)} catalysts for {ticker}")
            return validated_catalysts
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return catalysts
    
    def _are_catalysts_similar(self, catalyst1: Catalyst, catalyst2: Catalyst) -> bool:
        """Check if two catalysts are similar enough to cross-validate"""
        try:
            # Same category
            if catalyst1.category != catalyst2.category:
                return False
            
            # Similar impact range (within 20 points)
            if abs(catalyst1.impact - catalyst2.impact) > 20:
                return False
            
            # Similar sentiment
            if catalyst1.sentiment_label != catalyst2.sentiment_label:
                return False
            
            # Check for keyword overlap in titles/content
            text1 = f"{catalyst1.catalyst} {getattr(catalyst1, 'extra_data', {}).get('title', '')}"
            text2 = f"{catalyst2.catalyst} {getattr(catalyst2, 'extra_data', {}).get('title', '')}"
            
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate word overlap
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            
            similarity = overlap / total_unique if total_unique > 0 else 0
            
            return similarity > 0.3  # 30% word overlap threshold
            
        except Exception as e:
            logger.error(f"Error comparing catalysts: {e}")
            return False
    
    def _validate_catalyst_group(self, similar_catalysts: List[Catalyst]) -> Optional[Catalyst]:
        """Create validated catalyst from group of similar catalysts"""
        try:
            if not similar_catalysts:
                return None
            
            # Use the catalyst with highest confidence as base
            base_catalyst = max(similar_catalysts, key=lambda c: c.confidence)
            
            # Calculate weighted averages
            total_confidence = sum(c.confidence for c in similar_catalysts)
            avg_confidence = total_confidence / len(similar_catalysts)
            
            # Boost confidence for cross-validation
            validated_confidence = min(1.0, avg_confidence + 0.1)
            
            # Average impact scores
            avg_impact = sum(c.impact for c in similar_catalysts) / len(similar_catalysts)
            
            # Count sources
            sources = set(c.source for c in similar_catalysts)
            
            # Create validated catalyst
            validated = Catalyst(
                ticker=base_catalyst.ticker,
                catalyst=base_catalyst.catalyst,
                category=base_catalyst.category,
                sentiment_label=base_catalyst.sentiment_label,
                sentiment_score=base_catalyst.sentiment_score,
                impact=avg_impact,
                confidence=validated_confidence,
                source=f"Cross-validated ({len(sources)} sources)",
                published_date=base_catalyst.published_date,
                url=base_catalyst.url,
                extra_data={
                    **getattr(base_catalyst, 'extra_data', {}),
                    'cross_validated': True,
                    'source_count': len(sources),
                    'validation_sources': list(sources)
                }
            )
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating catalyst group: {e}")
            return similar_catalysts[0] if similar_catalysts else None

    async def analyze_trending_catalysts(self, time_window_hours: int = 24, 
                                       min_mentions: int = 3) -> List[Dict[str, Any]]:
        """Analyze trending catalysts and topics"""
        try:
            logger.info(f"Analyzing trending catalysts (last {time_window_hours}h, min {min_mentions} mentions)")
            
            # Get recent catalysts
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
            recent_catalysts = self.db_manager.get_catalysts(limit=1000)
            
            # Filter by time window
            time_filtered = [
                c for c in recent_catalysts
                if c.published_date and c.published_date >= cutoff_time
            ]
            
            # Analyze trending topics
            trending_topics = self._analyze_trending_topics(time_filtered, min_mentions)
            
            # Analyze trending tickers
            trending_tickers = self._analyze_trending_tickers(time_filtered, min_mentions)
            
            results = {
                'trending_topics': trending_topics,
                'trending_tickers': trending_tickers,
                'analysis_time': datetime.now(timezone.utc).isoformat(),
                'time_window_hours': time_window_hours,
                'total_catalysts_analyzed': len(time_filtered)
            }
            
            logger.info(f"Trending analysis completed: {len(trending_topics)} topics, {len(trending_tickers)} tickers")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing trending catalysts: {e}")
            return []
    
    def _analyze_trending_topics(self, catalysts: List[Catalyst], min_mentions: int) -> List[Dict[str, Any]]:
        """Analyze trending topics from catalyst text"""
        try:
            # Extract keywords from catalyst text
            keyword_counts = {}
            
            for catalyst in catalysts:
                # Simple keyword extraction (in production, use more sophisticated NLP)
                words = catalyst.catalyst.lower().split()
                
                # Filter for meaningful words (basic approach)
                meaningful_words = [
                    word for word in words
                    if len(word) > 3 and word.isalpha()
                    and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been']
                ]
                
                for word in meaningful_words:
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1
            
            # Filter and sort trending topics
            trending = [
                {'topic': topic, 'mentions': count, 'relevance_score': count / len(catalysts)}
                for topic, count in keyword_counts.items()
                if count >= min_mentions
            ]
            
            trending.sort(key=lambda x: x['mentions'], reverse=True)
            
            return trending[:20]  # Top 20 trending topics
            
        except Exception as e:
            logger.error(f"Error analyzing trending topics: {e}")
            return []
    
    def _analyze_trending_tickers(self, catalysts: List[Catalyst], min_mentions: int) -> List[Dict[str, Any]]:
        """Analyze trending tickers"""
        try:
            ticker_stats = {}
            
            for catalyst in catalysts:
                ticker = catalyst.ticker
                if ticker not in ticker_stats:
                    ticker_stats[ticker] = {
                        'ticker': ticker,
                        'mention_count': 0,
                        'avg_impact': 0,
                        'avg_sentiment': 0,
                        'categories': set()
                    }
                
                stats = ticker_stats[ticker]
                stats['mention_count'] += 1
                stats['avg_impact'] += catalyst.impact
                stats['avg_sentiment'] += catalyst.sentiment_score
                stats['categories'].add(catalyst.category.value)
            
            # Calculate averages and filter
            trending_tickers = []
            
            for ticker, stats in ticker_stats.items():
                if stats['mention_count'] >= min_mentions:
                    trending_tickers.append({
                        'ticker': ticker,
                        'mention_count': stats['mention_count'],
                        'avg_impact': stats['avg_impact'] / stats['mention_count'],
                        'avg_sentiment': stats['avg_sentiment'] / stats['mention_count'],
                        'categories': list(stats['categories']),
                        'trend_score': stats['mention_count'] * (stats['avg_impact'] / stats['mention_count'])
                    })
            
            # Sort by trend score
            trending_tickers.sort(key=lambda x: x['trend_score'], reverse=True)
            
            return trending_tickers[:10]  # Top 10 trending tickers
            
        except Exception as e:
            logger.error(f"Error analyzing trending tickers: {e}")
            return []
    
    async def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data from database and caches"""
        try:
            logger.info(f"Starting cleanup of data older than {days_old} days")
            
            # Clean up database
            self.db_manager.cleanup_old_records(days_old)
            
            # Clear old caches
            self.cache_manager.clear(namespace="batch_processing")
            
            # Clear fetcher caches
            fetchers = get_all_fetchers(self.config)
            for fetcher in fetchers.values():
                fetcher.clear_cache()
            
            logger.info(f"Cleanup completed for data older than {days_old} days")
            
            return days_old  # Return days cleaned for reporting
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    async def clear_expired_caches(self) -> int:
        """Clear expired caches"""
        try:
            # This would clear expired caches across the system
            # For now, just clear the batch processing cache
            self.cache_manager.clear(namespace="batch_processing")
            
            logger.info("Cleared expired caches")
            return 1  # Return count of caches cleared
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            return 0
    
    def _update_stats(self, items_processed: int, catalysts_generated: int, processing_time: float):
        """Update processing statistics"""
        self._stats['batches_processed'] += 1
        self._stats['total_items_processed'] += items_processed
        self._stats['catalysts_generated'] += catalysts_generated
        self._stats['processing_time_total'] += processing_time
        self._stats['last_batch_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate average processing time
        if self._stats['batches_processed'] > 0:
            self._stats['avg_processing_time'] = (
                self._stats['processing_time_total'] / self._stats['batches_processed']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return dict(self._stats)
    
    async def cleanup(self):
        """Cleanup processor resources"""
        try:
            await self.nlp_processor.cleanup()
            logger.info("Batch Processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during batch processor cleanup: {e}")
