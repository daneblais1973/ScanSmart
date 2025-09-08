import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import json
import os

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class EarningsFetcher(BaseFetcher):
    """Earnings call transcripts and earnings data fetcher"""
    
    def __init__(self):
        super().__init__()
        # Using Financial Modeling Prep for earnings data
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        # Alpha Vantage also provides earnings data
        self.av_base_url = "https://www.alphavantage.co/query"
        self._session = None
        self._session_lock = asyncio.Lock()
        
        logger.info("Earnings Fetcher initialized")
    
    def get_source_name(self) -> str:
        return "Earnings Data"
    
    def is_configured(self) -> bool:
        """Check if earnings APIs are configured"""
        fmp_key = os.getenv('FMP_API_KEY')
        av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        return fmp_key is not None or av_key is not None
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get available API keys"""
        return {
            'fmp': os.getenv('FMP_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY')
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-Earnings/1.0 (Earnings Analysis)'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch earnings data and transcripts"""
        try:
            # Get earnings calendar and historical earnings
            earnings_data = await self.get_earnings_data(ticker, limit//2)
            
            # Get earnings transcripts if available
            transcripts = await self.get_earnings_transcripts(ticker, limit//2)
            
            all_data = earnings_data + transcripts
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in Earnings fetch_all: {e}")
            return []
    
    async def get_earnings_data(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get earnings calendar and historical earnings data"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("earnings_data", ticker=ticker, limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            api_keys = self.get_api_keys()
            all_earnings = []
            
            # Try Financial Modeling Prep first
            if api_keys['fmp']:
                fmp_earnings = await self._fetch_fmp_earnings(session, ticker, api_keys['fmp'], limit)
                all_earnings.extend(fmp_earnings)
            
            # Try Alpha Vantage if we need more data
            if len(all_earnings) < limit and api_keys['alpha_vantage']:
                av_earnings = await self._fetch_av_earnings(session, ticker, api_keys['alpha_vantage'], limit)
                all_earnings.extend(av_earnings)
            
            # Cache results
            self.set_cache(cache_key, all_earnings)
            
            logger.info(f"Earnings: Fetched {len(all_earnings)} earnings records for {ticker}")
            return all_earnings[:limit]
            
        except Exception as e:
            logger.error(f"Error getting earnings data: {e}")
            return []
    
    async def _fetch_fmp_earnings(self, session: aiohttp.ClientSession, 
                                  ticker: str, api_key: str, limit: int) -> List[Dict]:
        """Fetch earnings data from Financial Modeling Prep"""
        earnings = []
        
        try:
            # Get earnings calendar
            url = f"{self.fmp_base_url}/earning_calendar"
            params = {
                'from': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'apikey': api_key
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter for the specific ticker
                    ticker_earnings = [item for item in data if item.get('symbol') == ticker.upper()]
                    
                    for item in ticker_earnings[:limit]:
                        earnings_data = {
                            'type': 'earnings_calendar',
                            'symbol': item.get('symbol'),
                            'date': item.get('date'),
                            'eps_estimated': item.get('epsEstimated'),
                            'eps_actual': item.get('epsActual'),
                            'revenue_estimated': item.get('revenueEstimated'),
                            'revenue_actual': item.get('revenueActual'),
                            'time': item.get('time'),
                            'updated_from_date': item.get('updatedFromDate'),
                            'source': 'Financial Modeling Prep',
                            'timestamp': datetime.now(timezone.utc),
                            'raw_data': item
                        }
                        
                        normalized = self.normalize_data_item(earnings_data, f"{ticker}_{item.get('date')}")
                        earnings.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                elif response.status == 429:
                    logger.warning("FMP API rate limit exceeded")
                    self._stats['failed_requests'] += 1
                else:
                    logger.warning(f"FMP earnings returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching FMP earnings: {e}")
            self._stats['failed_requests'] += 1
        
        return earnings
    
    async def _fetch_av_earnings(self, session: aiohttp.ClientSession, 
                                 ticker: str, api_key: str, limit: int) -> List[Dict]:
        """Fetch earnings data from Alpha Vantage"""
        earnings = []
        
        try:
            url = self.av_base_url
            params = {
                'function': 'EARNINGS',
                'symbol': ticker,
                'apikey': api_key
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'quarterlyEarnings' in data:
                        quarterly_earnings = data['quarterlyEarnings'][:limit]
                        
                        for item in quarterly_earnings:
                            earnings_data = {
                                'type': 'quarterly_earnings',
                                'symbol': ticker.upper(),
                                'fiscal_date_ending': item.get('fiscalDateEnding'),
                                'reported_date': item.get('reportedDate'),
                                'reported_eps': item.get('reportedEPS'),
                                'estimated_eps': item.get('estimatedEPS'),
                                'surprise': item.get('surprise'),
                                'surprise_percentage': item.get('surprisePercentage'),
                                'source': 'Alpha Vantage',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(earnings_data, f"{ticker}_{item.get('fiscalDateEnding')}")
                            earnings.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"Alpha Vantage earnings returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching Alpha Vantage earnings: {e}")
            self._stats['failed_requests'] += 1
        
        return earnings
    
    async def get_earnings_transcripts(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Get earnings call transcripts (using FMP if available)"""
        try:
            api_keys = self.get_api_keys()
            if not api_keys['fmp']:
                return []
            
            session = await self._get_session()
            transcripts = []
            
            # Get earnings transcripts from FMP
            url = f"{self.fmp_base_url}/earning_call_transcript/{ticker.upper()}"
            params = {
                'apikey': api_keys['fmp']
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        for item in data[:limit]:
                            transcript_data = {
                                'type': 'earnings_transcript',
                                'symbol': ticker.upper(),
                                'quarter': item.get('quarter'),
                                'year': item.get('year'),
                                'date': item.get('date'),
                                'content': item.get('content', '')[:2000] + '...' if item.get('content') else None,
                                'source': 'Financial Modeling Prep',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(transcript_data, f"{ticker}_transcript_{item.get('quarter')}_{item.get('year')}")
                            transcripts.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"FMP transcripts returned status {response.status}")
                    self._stats['failed_requests'] += 1
            
            return transcripts
            
        except Exception as e:
            logger.error(f"Error getting earnings transcripts: {e}")
            return []
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test earnings API connections"""
        try:
            api_keys = self.get_api_keys()
            
            if not any(api_keys.values()):
                return {
                    'success': False,
                    'message': "No earnings API keys configured (FMP_API_KEY, ALPHA_VANTAGE_API_KEY)",
                    'fetcher': 'Earnings Data'
                }
            
            session = await self._get_session()
            working_apis = []
            
            # Test FMP
            if api_keys['fmp']:
                try:
                    url = f"{self.fmp_base_url}/earning_calendar"
                    params = {
                        'from': datetime.now().strftime('%Y-%m-%d'),
                        'to': datetime.now().strftime('%Y-%m-%d'),
                        'apikey': api_keys['fmp']
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            working_apis.append('FMP')
                except:
                    pass
            
            # Test Alpha Vantage
            if api_keys['alpha_vantage']:
                try:
                    url = self.av_base_url
                    params = {
                        'function': 'EARNINGS',
                        'symbol': 'AAPL',
                        'apikey': api_keys['alpha_vantage']
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'quarterlyEarnings' in data:
                                working_apis.append('Alpha Vantage')
                except:
                    pass
            
            if working_apis:
                return {
                    'success': True,
                    'message': f"Earnings API connection successful ({', '.join(working_apis)})",
                    'fetcher': 'Earnings Data'
                }
            else:
                return {
                    'success': False,
                    'message': "All earnings API connections failed",
                    'fetcher': 'Earnings Data'
                }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Earnings connection failed: {str(e)}",
                'fetcher': 'Earnings Data'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()