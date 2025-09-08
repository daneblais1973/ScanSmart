import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
import os

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class EIAFetcher(BaseFetcher):
    """Energy Information Administration API fetcher - requires free API key"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.eia.gov/v2"
        self._session = None
        self._session_lock = asyncio.Lock()
        
        # Key energy datasets
        self.energy_datasets = {
            'CRUDE_OIL': 'petroleum/pri/spt/data',  # Spot prices
            'NATURAL_GAS': 'natural-gas/pri/sum/data',  # Natural gas prices
            'GASOLINE': 'petroleum/pri/gnd/data',  # Gasoline prices
            'COAL': 'coal/data',  # Coal data
            'ELECTRICITY': 'electricity/data',  # Electricity data
            'RENEWABLE': 'renewable/data',  # Renewable energy
            'NUCLEAR': 'nuclear-outages/data'  # Nuclear outages
        }
        
        logger.info("EIA Fetcher initialized")
    
    def get_source_name(self) -> str:
        return "EIA Energy"
    
    def is_configured(self) -> bool:
        """Check if EIA API key is configured"""
        return self.get_api_key() is not None
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        return os.getenv('EIA_API_KEY')
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-EIA/1.0 (Energy Market Analysis)'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch energy market data"""
        try:
            if not self.is_configured():
                logger.warning("EIA API key not configured")
                return []
            
            # Get energy prices and data
            energy_data = await self.get_energy_data(limit)
            return energy_data
            
        except Exception as e:
            logger.error(f"Error in EIA fetch_all: {e}")
            return []
    
    async def get_energy_data(self, limit: int = 20) -> List[Dict]:
        """Get energy market data"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("energy_data", limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            api_key = self.get_api_key()
            all_data = []
            
            # Get oil prices (WTI and Brent)
            oil_data = await self._fetch_oil_prices(session, api_key, limit//4)
            all_data.extend(oil_data)
            
            # Get natural gas prices
            gas_data = await self._fetch_natural_gas_prices(session, api_key, limit//4)
            all_data.extend(gas_data)
            
            # Get gasoline prices
            gasoline_data = await self._fetch_gasoline_prices(session, api_key, limit//4)
            all_data.extend(gasoline_data)
            
            # Get electricity data
            electric_data = await self._fetch_electricity_data(session, api_key, limit//4)
            all_data.extend(electric_data)
            
            # Cache results
            self.set_cache(cache_key, all_data)
            
            logger.info(f"EIA: Fetched {len(all_data)} energy data points")
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting EIA energy data: {e}")
            return []
    
    async def _fetch_oil_prices(self, session: aiohttp.ClientSession, 
                               api_key: str, limit: int) -> List[Dict]:
        """Fetch crude oil spot prices"""
        oil_data = []
        
        try:
            # WTI Crude Oil Spot Price
            url = f"{self.base_url}/petroleum/pri/spt/data"
            params = {
                'api_key': api_key,
                'frequency': 'daily',
                'data[0]': 'value',
                'facets[product][]': 'EPCCRUDE',  # WTI
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': limit
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'response' in data and 'data' in data['response']:
                        for item in data['response']['data']:
                            oil_point = {
                                'type': 'oil_price',
                                'product': 'WTI Crude Oil',
                                'price': item.get('value'),
                                'date': item.get('period'),
                                'units': item.get('units'),
                                'area': item.get('area-name'),
                                'source': 'EIA',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(oil_point, f"WTI_{item.get('period')}")
                            oil_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"EIA oil prices returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching oil prices: {e}")
            self._stats['failed_requests'] += 1
        
        return oil_data
    
    async def _fetch_natural_gas_prices(self, session: aiohttp.ClientSession, 
                                       api_key: str, limit: int) -> List[Dict]:
        """Fetch natural gas prices"""
        gas_data = []
        
        try:
            url = f"{self.base_url}/natural-gas/pri/sum/data"
            params = {
                'api_key': api_key,
                'frequency': 'monthly',
                'data[0]': 'value',
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': limit
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'response' in data and 'data' in data['response']:
                        for item in data['response']['data']:
                            gas_point = {
                                'type': 'natural_gas_price',
                                'price': item.get('value'),
                                'date': item.get('period'),
                                'units': item.get('units'),
                                'area': item.get('area-name'),
                                'source': 'EIA',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(gas_point, f"NATGAS_{item.get('period')}")
                            gas_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"EIA natural gas returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching natural gas prices: {e}")
            self._stats['failed_requests'] += 1
        
        return gas_data
    
    async def _fetch_gasoline_prices(self, session: aiohttp.ClientSession, 
                                    api_key: str, limit: int) -> List[Dict]:
        """Fetch gasoline prices"""
        gasoline_data = []
        
        try:
            url = f"{self.base_url}/petroleum/pri/gnd/data"
            params = {
                'api_key': api_key,
                'frequency': 'weekly',
                'data[0]': 'value',
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': limit
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'response' in data and 'data' in data['response']:
                        for item in data['response']['data']:
                            gasoline_point = {
                                'type': 'gasoline_price',
                                'price': item.get('value'),
                                'date': item.get('period'),
                                'units': item.get('units'),
                                'area': item.get('area-name'),
                                'source': 'EIA',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(gasoline_point, f"GASOLINE_{item.get('period')}")
                            gasoline_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"EIA gasoline prices returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching gasoline prices: {e}")
            self._stats['failed_requests'] += 1
        
        return gasoline_data
    
    async def _fetch_electricity_data(self, session: aiohttp.ClientSession, 
                                     api_key: str, limit: int) -> List[Dict]:
        """Fetch electricity generation data"""
        electric_data = []
        
        try:
            url = f"{self.base_url}/electricity/rto/region-data/data"
            params = {
                'api_key': api_key,
                'frequency': 'hourly',
                'data[0]': 'value',
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': limit
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'response' in data and 'data' in data['response']:
                        for item in data['response']['data']:
                            electric_point = {
                                'type': 'electricity_data',
                                'value': item.get('value'),
                                'date': item.get('period'),
                                'units': item.get('units'),
                                'respondent': item.get('respondent-name'),
                                'type_name': item.get('type-name'),
                                'source': 'EIA',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(electric_point, f"ELECTRIC_{item.get('period')}")
                            electric_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"EIA electricity data returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching electricity data: {e}")
            self._stats['failed_requests'] += 1
        
        return electric_data
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test EIA API connection"""
        try:
            if not self.is_configured():
                return {
                    'success': False,
                    'message': "EIA API key not configured (EIA_API_KEY)",
                    'fetcher': 'EIA Energy'
                }
            
            session = await self._get_session()
            api_key = self.get_api_key()
            
            # Test with a simple oil price request
            url = f"{self.base_url}/petroleum/pri/spt/data"
            params = {
                'api_key': api_key,
                'frequency': 'daily',
                'data[0]': 'value',
                'length': 1
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'response' in data:
                        return {
                            'success': True,
                            'message': "EIA API connection successful",
                            'fetcher': 'EIA Energy'
                        }
                elif response.status == 403:
                    return {
                        'success': False,
                        'message': "EIA API key invalid or expired",
                        'fetcher': 'EIA Energy'
                    }
            
            return {
                'success': False,
                'message': f"EIA API returned status {response.status}",
                'fetcher': 'EIA Energy'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"EIA connection failed: {str(e)}",
                'fetcher': 'EIA Energy'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()