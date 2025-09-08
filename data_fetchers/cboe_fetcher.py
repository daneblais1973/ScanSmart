import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
import os

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class CBOEFetcher(BaseFetcher):
    """CBOE (Chicago Board Options Exchange) data fetcher - VIX and options data"""
    
    def __init__(self):
        super().__init__()
        # CBOE offers some free market data feeds
        self.base_url = "https://www.cboe.com/api/global/delayed_quotes"
        self.vix_url = "https://cdn.cboe.com/api/global/delayed_quotes/indices/_VIX.json"
        self._session = None
        self._session_lock = asyncio.Lock()
        
        # CBOE indices to track
        self.cboe_indices = [
            'VIX',     # Volatility Index
            'VXN',     # NASDAQ-100 Volatility
            'RVX',     # Russell 2000 Volatility
            'VXO',     # S&P 100 Volatility
            'SKEW',    # CBOE Skew Index
            'PUT'      # CBOE Put/Call Ratio
        ]
        
        logger.info("CBOE Fetcher initialized - Free market data")
    
    def get_source_name(self) -> str:
        return "CBOE"
    
    def is_configured(self) -> bool:
        """CBOE free data doesn't require configuration"""
        return True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-CBOE/1.0 (Market Volatility Analysis)',
                    'Accept': 'application/json'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch CBOE volatility and options data"""
        try:
            # Get VIX and volatility indices
            vix_data = await self.get_vix_data(limit//2)
            
            # Get options data for the ticker if available
            options_data = await self.get_options_data(ticker, limit//2)
            
            all_data = vix_data + options_data
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in CBOE fetch_all: {e}")
            return []
    
    async def get_vix_data(self, limit: int = 10) -> List[Dict]:
        """Get VIX and other volatility indices"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("vix_data", limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            all_data = []
            
            # Fetch each CBOE index
            for index in self.cboe_indices:
                vix_point = await self._fetch_cboe_index(session, index)
                if vix_point:
                    all_data.append(vix_point)
                    
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Cache results
            self.set_cache(cache_key, all_data)
            
            logger.info(f"CBOE: Fetched {len(all_data)} volatility indices")
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return []
    
    async def _fetch_cboe_index(self, session: aiohttp.ClientSession, index: str) -> Optional[Dict]:
        """Fetch specific CBOE index data"""
        try:
            url = f"https://cdn.cboe.com/api/global/delayed_quotes/indices/_{index}.json"
            
            self._stats['total_requests'] += 1
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data:
                        index_data = data['data']
                        
                        vix_point = {
                            'type': 'volatility_index',
                            'symbol': index,
                            'name': index_data.get('name', index),
                            'current_price': index_data.get('current_price'),
                            'price_change': index_data.get('change'),
                            'percent_change': index_data.get('change_percent'),
                            'high': index_data.get('high'),
                            'low': index_data.get('low'),
                            'open': index_data.get('open'),
                            'previous_close': index_data.get('prev_close'),
                            'volume': index_data.get('volume'),
                            'updated_at': index_data.get('timestamp'),
                            'source': 'CBOE',
                            'timestamp': datetime.now(timezone.utc),
                            'raw_data': data
                        }
                        
                        normalized = self.normalize_data_item(vix_point, index)
                        
                        self._stats['successful_requests'] += 1
                        return normalized
                    
                else:
                    logger.warning(f"CBOE {index} returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching CBOE index {index}: {e}")
            self._stats['failed_requests'] += 1
        
        return None
    
    async def get_options_data(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get options data for a specific ticker"""
        try:
            session = await self._get_session()
            options_data = []
            
            # Try to get options quotes for the ticker
            # Note: CBOE's free API has limited options data
            url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker.upper()}.json"
            
            self._stats['total_requests'] += 1
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and 'options' in data['data']:
                        options = data['data']['options'][:limit]
                        
                        for option in options:
                            option_data = {
                                'type': 'options_data',
                                'underlying_symbol': ticker.upper(),
                                'option_symbol': option.get('option'),
                                'strike': option.get('strike'),
                                'expiration': option.get('expiration'),
                                'option_type': option.get('option_type'),  # call/put
                                'last_price': option.get('last'),
                                'bid': option.get('bid'),
                                'ask': option.get('ask'),
                                'volume': option.get('volume'),
                                'open_interest': option.get('open_interest'),
                                'implied_volatility': option.get('iv'),
                                'delta': option.get('delta'),
                                'gamma': option.get('gamma'),
                                'theta': option.get('theta'),
                                'vega': option.get('vega'),
                                'source': 'CBOE',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': option
                            }
                            
                            normalized = self.normalize_data_item(option_data, option.get('option', ''))
                            options_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                elif response.status == 404:
                    # Options data not available for this ticker
                    logger.debug(f"No options data available for {ticker}")
                    self._stats['failed_requests'] += 1
                else:
                    logger.warning(f"CBOE options returned status {response.status}")
                    self._stats['failed_requests'] += 1
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options data: {e}")
            return []
    
    async def get_put_call_ratio(self) -> Dict:
        """Get CBOE Put/Call ratio"""
        try:
            session = await self._get_session()
            
            url = "https://cdn.cboe.com/api/global/delayed_quotes/indices/_PUT.json"
            
            self._stats['total_requests'] += 1
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data:
                        ratio_data = {
                            'type': 'put_call_ratio',
                            'ratio': data['data'].get('current_price'),
                            'change': data['data'].get('change'),
                            'percent_change': data['data'].get('change_percent'),
                            'timestamp': datetime.now(timezone.utc),
                            'source': 'CBOE',
                            'raw_data': data
                        }
                        
                        self._stats['successful_requests'] += 1
                        return ratio_data
                    
                else:
                    logger.warning(f"CBOE put/call ratio returned status {response.status}")
                    self._stats['failed_requests'] += 1
                    return {}
        
        except Exception as e:
            logger.error(f"Error getting put/call ratio: {e}")
            return {}
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test CBOE API connection"""
        try:
            session = await self._get_session()
            
            # Test with VIX data
            url = "https://cdn.cboe.com/api/global/delayed_quotes/indices/_VIX.json"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        return {
                            'success': True,
                            'message': "CBOE API connection successful",
                            'fetcher': 'CBOE'
                        }
            
            return {
                'success': False,
                'message': f"CBOE API returned status {response.status}",
                'fetcher': 'CBOE'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"CBOE connection failed: {str(e)}",
                'fetcher': 'CBOE'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()