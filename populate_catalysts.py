#!/usr/bin/env python3
"""
Populate catalyst database with live data from available sources
Run this to populate the database with real catalyst data for testing
"""

import asyncio
import logging
from datetime import datetime, timezone
from core.config import AppConfig
from core.database import DatabaseManager
from data_fetchers.rss_fetcher import RSSFetcher
from nlp.rule_based_detector import RuleBasedCatalystDetector
from nlp.processor import NLPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def populate_catalysts():
    """Populate database with live catalyst data"""
    
    logger.info("Starting catalyst population...")
    
    # Initialize components
    config = AppConfig()
    db = DatabaseManager(config)
    rss_fetcher = RSSFetcher()
    catalyst_detector = RuleBasedCatalystDetector()
    nlp_processor = NLPProcessor(config)
    
    try:
        # Test popular tickers
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']
        
        catalysts_added = 0
        
        for ticker in test_tickers:
            try:
                logger.info(f"Fetching data for {ticker}...")
                
                # Get RSS data for ticker
                rss_data = await rss_fetcher.fetch_all(ticker, limit=10)
                
                for item in rss_data:
                    try:
                        # Extract text content
                        text_content = item.get('title', '') + ' ' + item.get('description', '')
                        
                        if len(text_content.strip()) < 10:
                            continue
                            
                        # Detect catalysts using rule-based detector
                        catalysts = catalyst_detector.detect_catalysts(text_content, ticker)
                        
                        for catalyst_dict in catalysts:
                            try:
                                # Create catalyst object
                                from core.models import Catalyst, CatalystCategory, SentimentLabel, DataSource
                                
                                catalyst = Catalyst(
                                    ticker=ticker,
                                    catalyst=text_content[:500],  # Limit length
                                    category=CatalystCategory(catalyst_dict.get('category', 'news')),
                                    sentiment_label=SentimentLabel(catalyst_dict.get('sentiment', 'neutral')),
                                    sentiment_score=catalyst_dict.get('sentiment_score', 0.0),
                                    impact=min(100, max(1, int(catalyst_dict.get('impact_score', 50)))),
                                    source=DataSource.RSS,
                                    confidence=catalyst_dict.get('confidence', 0.5),
                                    published_date=datetime.now(timezone.utc),
                                    url=item.get('link', ''),
                                    sector='Technology',  # Default sector
                                    metadata={'source_item': item.get('title', '')[:100]}
                                )
                                
                                # Save to database
                                catalyst_id = db.save_catalyst(catalyst)
                                if catalyst_id:
                                    catalysts_added += 1
                                    logger.info(f"Added catalyst {catalysts_added}: {ticker} - {text_content[:50]}...")
                                    
                            except Exception as e:
                                logger.warning(f"Error creating catalyst for {ticker}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Error processing RSS item for {ticker}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error fetching data for {ticker}: {e}")
                continue
                
            # Add small delay between tickers
            await asyncio.sleep(1)
        
        logger.info(f"✅ Successfully added {catalysts_added} catalysts to database")
        
        # Show final count
        total_catalysts = db.get_catalyst_count()
        logger.info(f"📊 Total catalysts in database: {total_catalysts}")
        
    except Exception as e:
        logger.error(f"Error in catalyst population: {e}")
    finally:
        # Cleanup
        await rss_fetcher.cleanup()

if __name__ == "__main__":
    asyncio.run(populate_catalysts())