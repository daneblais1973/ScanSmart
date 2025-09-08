import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import json
import xml.etree.ElementTree as ET

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class ECBFetcher(BaseFetcher):
    """European Central Bank Statistical Data Warehouse fetcher - free access"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://data-api.ecb.europa.eu/service/data"
        self._session = None
        self._session_lock = asyncio.Lock()
        
        # Key ECB datasets
        self.key_indicators = {
            'POLICY_RATES': 'FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA',  # Policy rates
            'EXCHANGE_RATES': 'EXR/D.USD.EUR.SP00.A',  # EUR/USD exchange rate
            'INFLATION': 'ICP/M.U2.N.000000.4.ANR',  # HICP inflation
            'GDP': 'MNA/Q.N.I8.W1.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N',  # GDP
            'MONEY_SUPPLY': 'BSI/M.U2.N.A.A20.A.1.U2.2240.Z01.E',  # M1 money supply
            'UNEMPLOYMENT': 'LFSI/M.I8.S.UNEHRT.TOTAL0.15_74.T'  # Unemployment rate
        }
        
        logger.info("ECB Fetcher initialized - No API key required")
    
    def get_source_name(self) -> str:
        return "European Central Bank"
    
    def is_configured(self) -> bool:
        """ECB data doesn't require configuration"""
        return True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-ECB/1.0 (European Economic Analysis)',
                    'Accept': 'application/json, application/xml'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch ECB economic indicators"""
        try:
            # Get key European economic indicators
            indicators = await self.get_ecb_indicators(limit)
            return indicators
            
        except Exception as e:
            logger.error(f"Error in ECB fetch_all: {e}")
            return []
    
    async def get_ecb_indicators(self, limit: int = 20) -> List[Dict]:
        """Get ECB key economic indicators"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("ecb_indicators", limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            all_data = []
            
            # Fetch each key indicator
            for indicator_name, dataset_key in self.key_indicators.items():
                indicator_data = await self._fetch_ecb_dataset(session, indicator_name, dataset_key)
                if indicator_data:
                    all_data.extend(indicator_data[:5])  # Latest 5 points per indicator
                    
                # Rate limiting
                await asyncio.sleep(1)
                
                if len(all_data) >= limit:
                    break
            
            # Cache results
            self.set_cache(cache_key, all_data)
            
            logger.info(f"ECB: Fetched {len(all_data)} economic indicators")
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting ECB indicators: {e}")
            return []
    
    async def _fetch_ecb_dataset(self, session: aiohttp.ClientSession, 
                                 indicator_name: str, dataset_key: str) -> List[Dict]:
        """Fetch specific ECB dataset"""
        data_points = []
        
        try:
            # ECB API URL with recent data filter
            url = f"{self.base_url}/{dataset_key}"
            params = {
                'startPeriod': (datetime.now() - timedelta(days=365)).strftime('%Y-%m'),
                'endPeriod': datetime.now().strftime('%Y-%m'),
                'format': 'jsondata'
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # ECB JSON structure: dataSets[0].series
                    if 'dataSets' in data and len(data['dataSets']) > 0:
                        series = data['dataSets'][0].get('series', {})
                        
                        # Get structure for time periods
                        structure = data.get('structure', {})
                        dimensions = structure.get('dimensions', {}).get('observation', [])
                        time_dimension = None
                        
                        for dim in dimensions:
                            if dim.get('id') == 'TIME_PERIOD':
                                time_dimension = dim.get('values', [])
                                break
                        
                        # Process series data
                        for series_key, series_data in series.items():
                            observations = series_data.get('observations', {})
                            
                            for obs_key, obs_value in observations.items():
                                try:
                                    time_index = int(obs_key)
                                    if time_dimension and time_index < len(time_dimension):
                                        time_period = time_dimension[time_index].get('id')
                                        value = obs_value[0] if isinstance(obs_value, list) else obs_value
                                        
                                        if value is not None:
                                            data_point = {
                                                'type': 'ecb_indicator',
                                                'indicator': indicator_name,
                                                'dataset_key': dataset_key,
                                                'time_period': time_period,
                                                'value': float(value) if isinstance(value, (int, float)) else value,
                                                'source': 'ECB',
                                                'timestamp': datetime.now(timezone.utc),
                                            }
                                            
                                            normalized = self.normalize_data_item(data_point, f"{indicator_name}_{time_period}")
                                            data_points.append(normalized)
                                            
                                except (ValueError, IndexError, KeyError):
                                    continue
                    
                    self._stats['successful_requests'] += 1
                    
                elif response.status == 404:
                    logger.debug(f"ECB dataset {dataset_key} not found")
                    self._stats['failed_requests'] += 1
                else:
                    logger.warning(f"ECB API returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching ECB dataset {dataset_key}: {e}")
            self._stats['failed_requests'] += 1
        
        return data_points
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test ECB API connection"""
        try:
            session = await self._get_session()
            
            # Test with EUR/USD exchange rate
            url = f"{self.base_url}/EXR/D.USD.EUR.SP00.A"
            params = {
                'lastNObservations': 1,
                'format': 'jsondata'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'dataSets' in data:
                        return {
                            'success': True,
                            'message': "ECB API connection successful",
                            'fetcher': 'European Central Bank'
                        }
            
            return {
                'success': False,
                'message': f"ECB API returned status {response.status}",
                'fetcher': 'European Central Bank'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"ECB connection failed: {str(e)}",
                'fetcher': 'European Central Bank'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()