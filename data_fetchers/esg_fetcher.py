import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
import os

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class ESGFetcher(BaseFetcher):
    """ESG (Environmental, Social, Governance) data fetcher"""
    
    def __init__(self):
        super().__init__()
        # Using multiple ESG data sources
        self.sustain_base_url = "https://api.sustainalytics.com/v1"
        self.msci_base_url = "https://api.msci.com/esg/v2.4"
        self._session = None
        self._session_lock = asyncio.Lock()
        
        # ESG metrics to track
        self.esg_metrics = [
            'carbon_footprint',
            'water_usage',
            'waste_management',
            'renewable_energy',
            'employee_satisfaction',
            'board_diversity',
            'executive_compensation',
            'data_privacy',
            'supply_chain_ethics'
        ]
        
        logger.info("ESG Fetcher initialized")
    
    def get_source_name(self) -> str:
        return "ESG Data"
    
    def is_configured(self) -> bool:
        """Check if ESG API keys are configured"""
        sustainalytics_key = os.getenv('SUSTAINALYTICS_API_KEY')
        msci_key = os.getenv('MSCI_ESG_API_KEY')
        return sustainalytics_key is not None or msci_key is not None
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get available API keys"""
        return {
            'sustainalytics': os.getenv('SUSTAINALYTICS_API_KEY'),
            'msci': os.getenv('MSCI_ESG_API_KEY')
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-ESG/1.0 (Sustainability Analysis)'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch ESG data for a company"""
        try:
            if not self.is_configured():
                logger.warning("No ESG API keys configured")
                return []
            
            # Get ESG ratings and metrics
            esg_data = await self.get_esg_data(ticker, limit//2)
            
            # Get sustainability trends
            sustainability_data = await self.get_sustainability_trends(limit//2)
            
            all_data = esg_data + sustainability_data
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in ESG fetch_all: {e}")
            return []
    
    async def get_esg_data(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get ESG ratings and scores for a company"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("esg_data", ticker=ticker, limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            api_keys = self.get_api_keys()
            all_esg_data = []
            
            # Try Sustainalytics first
            if api_keys['sustainalytics']:
                sustainalytics_data = await self._fetch_sustainalytics_data(session, ticker, api_keys['sustainalytics'], limit)
                all_esg_data.extend(sustainalytics_data)
            
            # Try MSCI ESG if we need more data
            if len(all_esg_data) < limit and api_keys['msci']:
                msci_data = await self._fetch_msci_esg_data(session, ticker, api_keys['msci'], limit)
                all_esg_data.extend(msci_data)
            
            # Cache results
            self.set_cache(cache_key, all_esg_data)
            
            logger.info(f"ESG: Fetched {len(all_esg_data)} ESG records for {ticker}")
            return all_esg_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting ESG data: {e}")
            return []
    
    async def _fetch_sustainalytics_data(self, session: aiohttp.ClientSession, 
                                        ticker: str, api_key: str, limit: int) -> List[Dict]:
        """Fetch ESG data from Sustainalytics"""
        esg_data = []
        
        try:
            # Sustainalytics company ESG risk rating
            url = f"{self.sustain_base_url}/companies/{ticker}/esg-risk-rating"
            headers = {'Authorization': f'Bearer {api_key}'}
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    esg_rating = {
                        'type': 'esg_rating',
                        'symbol': ticker.upper(),
                        'provider': 'Sustainalytics',
                        'esg_risk_score': data.get('esgRiskScore'),
                        'esg_risk_category': data.get('esgRiskCategory'),
                        'environment_score': data.get('environmentScore'),
                        'social_score': data.get('socialScore'),
                        'governance_score': data.get('governanceScore'),
                        'controversy_score': data.get('controversyScore'),
                        'industry_rank': data.get('industryRank'),
                        'percentile_rank': data.get('percentileRank'),
                        'last_updated': data.get('lastUpdated'),
                        'source': 'Sustainalytics',
                        'timestamp': datetime.now(timezone.utc),
                        'raw_data': data
                    }
                    
                    normalized = self.normalize_data_item(esg_rating, f"{ticker}_sustainalytics")
                    esg_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                elif response.status == 404:
                    logger.debug(f"No Sustainalytics ESG data for {ticker}")
                    self._stats['failed_requests'] += 1
                else:
                    logger.warning(f"Sustainalytics ESG returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching Sustainalytics data: {e}")
            self._stats['failed_requests'] += 1
        
        return esg_data
    
    async def _fetch_msci_esg_data(self, session: aiohttp.ClientSession, 
                                  ticker: str, api_key: str, limit: int) -> List[Dict]:
        """Fetch ESG data from MSCI"""
        esg_data = []
        
        try:
            url = f"{self.msci_base_url}/companies/{ticker}/esg-scores"
            headers = {'X-API-Key': api_key}
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'esgScores' in data:
                        for score_data in data['esgScores']:
                            esg_score = {
                                'type': 'msci_esg_score',
                                'symbol': ticker.upper(),
                                'provider': 'MSCI',
                                'esg_rating': score_data.get('esgRating'),
                                'esg_score': score_data.get('esgScore'),
                                'environment_score': score_data.get('environmentScore'),
                                'social_score': score_data.get('socialScore'),
                                'governance_score': score_data.get('governanceScore'),
                                'industry': score_data.get('industry'),
                                'peer_group': score_data.get('peerGroup'),
                                'date': score_data.get('asOfDate'),
                                'source': 'MSCI ESG',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': score_data
                            }
                            
                            normalized = self.normalize_data_item(esg_score, f"{ticker}_msci_{score_data.get('asOfDate')}")
                            esg_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    
                else:
                    logger.warning(f"MSCI ESG returned status {response.status}")
                    self._stats['failed_requests'] += 1
        
        except Exception as e:
            logger.warning(f"Error fetching MSCI ESG data: {e}")
            self._stats['failed_requests'] += 1
        
        return esg_data
    
    async def get_sustainability_trends(self, limit: int = 10) -> List[Dict]:
        """Get general sustainability and ESG trends"""
        try:
            # For demonstration, we'll create sample trend data
            # In practice, this would connect to sustainability databases
            trends = []
            
            trend_topics = [
                'carbon_neutrality_commitments',
                'renewable_energy_adoption',
                'sustainable_investing_growth',
                'esg_regulation_updates',
                'climate_risk_disclosure'
            ]
            
            for i, topic in enumerate(trend_topics[:limit]):
                trend = {
                    'type': 'sustainability_trend',
                    'topic': topic,
                    'trend_direction': 'increasing' if i % 2 == 0 else 'stable',
                    'impact_score': 7.5 + (i * 0.3),  # Sample scoring
                    'description': f"Market trend analysis for {topic.replace('_', ' ')}",
                    'source': 'ESG Research',
                    'timestamp': datetime.now(timezone.utc),
                    'raw_data': {'topic': topic, 'analysis': 'trend_analysis'}
                }
                
                normalized = self.normalize_data_item(trend, f"trend_{topic}")
                trends.append(normalized)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting sustainability trends: {e}")
            return []
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test ESG API connections"""
        try:
            api_keys = self.get_api_keys()
            
            if not any(api_keys.values()):
                return {
                    'success': False,
                    'message': "No ESG API keys configured (SUSTAINALYTICS_API_KEY, MSCI_ESG_API_KEY)",
                    'fetcher': 'ESG Data'
                }
            
            session = await self._get_session()
            working_apis = []
            
            # Test Sustainalytics
            if api_keys['sustainalytics']:
                try:
                    url = f"{self.sustain_base_url}/health"
                    headers = {'Authorization': f'Bearer {api_keys["sustainalytics"]}'}
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            working_apis.append('Sustainalytics')
                except:
                    pass
            
            # Test MSCI
            if api_keys['msci']:
                try:
                    url = f"{self.msci_base_url}/health"
                    headers = {'X-API-Key': api_keys['msci']}
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            working_apis.append('MSCI')
                except:
                    pass
            
            if working_apis:
                return {
                    'success': True,
                    'message': f"ESG API connection successful ({', '.join(working_apis)})",
                    'fetcher': 'ESG Data'
                }
            else:
                return {
                    'success': False,
                    'message': "All ESG API connections failed",
                    'fetcher': 'ESG Data'
                }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"ESG connection failed: {str(e)}",
                'fetcher': 'ESG Data'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()