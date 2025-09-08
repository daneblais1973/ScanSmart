import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class FinancialFetcher(BaseFetcher):
    """Financial data fetcher for market data and company information"""
    
    def __init__(self, fmp_key: Optional[str] = None, alpha_vantage_key: Optional[str] = None, polygon_key: Optional[str] = None):
        super().__init__()
        
        # API keys for various financial data providers
        self.fmp_key = fmp_key  # Financial Modeling Prep
        self.alpha_vantage_key = alpha_vantage_key  # Alpha Vantage
        self.polygon_key = polygon_key  # Polygon.io
        
        # API endpoints
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        self.polygon_base_url = "https://api.polygon.io"
        
        self._session = None
        self._session_lock = asyncio.Lock()
        
        logger.info("Financial Fetcher initialized")
    
    def get_source_name(self) -> str:
        return "Financial"
    
    def is_configured(self) -> bool:
        """Check if at least one financial API is configured"""
        return any([self.fmp_key, self.alpha_vantage_key, self.polygon_key])
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch financial data for a ticker"""
        try:
            validated_ticker = self.validate_ticker(ticker)
            
            if not self.is_configured():
                logger.warning("No financial APIs configured")
                return []
            
            # Check cache first
            cache_key = self.generate_cache_key("fetch_all", ticker=validated_ticker, limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Collect data from all available sources
            all_data = []
            
            # Fetch from Financial Modeling Prep
            if self.fmp_key:
                try:
                    fmp_data = await self._fetch_fmp_data(validated_ticker, limit)
                    all_data.extend(fmp_data)
                except Exception as e:
                    logger.warning(f"FMP data fetch failed: {e}")
            
            # Fetch from Alpha Vantage
            if self.alpha_vantage_key:
                try:
                    av_data = await self._fetch_alpha_vantage_data(validated_ticker, limit)
                    all_data.extend(av_data)
                except Exception as e:
                    logger.warning(f"Alpha Vantage data fetch failed: {e}")
            
            # Fetch from Polygon
            if self.polygon_key:
                try:
                    polygon_data = await self._fetch_polygon_data(validated_ticker, limit)
                    all_data.extend(polygon_data)
                except Exception as e:
                    logger.warning(f"Polygon data fetch failed: {e}")
            
            # PROFESSIONAL MODE: No placeholder data - only return real market data
            if not all_data and not any([self.fmp_key, self.alpha_vantage_key, self.polygon_key]):
                logger.warning(f"No financial APIs configured and no data found for {validated_ticker}")
                # Return empty list instead of fake data for professional use
                all_data = []
            
            # Normalize all data
            normalized_data = []
            for item in all_data:
                try:
                    normalized = self.normalize_data_item(item, validated_ticker)
                    normalized_data.append(normalized)
                except Exception as e:
                    logger.warning(f"Failed to normalize financial data: {e}")
                    continue
            
            # Cache results
            self.set_cache(cache_key, normalized_data)
            
            logger.info(f"Financial: Fetched {len(normalized_data)} items for {validated_ticker}")
            return normalized_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in Financial fetch_all: {e}")
            return []
    
    async def _fetch_fmp_data(self, ticker: str, limit: int) -> List[Dict]:
        """Fetch data from Financial Modeling Prep"""
        session = await self._get_session()
        data = []
        
        try:
            # Company profile
            url = f"{self.fmp_base_url}/profile/{ticker}"
            params = {'apikey': self.fmp_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    profile_data = await response.json()
                    if profile_data:
                        company = profile_data[0] if isinstance(profile_data, list) else profile_data
                        data.append({
                            'type': 'company_profile',
                            'title': f'Company Profile - {company.get("companyName", ticker)}',
                            'content': company.get('description', ''),
                            'url': company.get('website', ''),
                            'sector': company.get('sector', ''),
                            'industry': company.get('industry', ''),
                            'market_cap': company.get('mktCap', 0),
                            'source': 'Financial Modeling Prep',
                            'raw_data': company
                        })
            
            # Recent news
            news_url = f"{self.fmp_base_url}/stock_news"
            news_params = {'apikey': self.fmp_key, 'tickers': ticker, 'limit': min(limit, 50)}
            
            async with session.get(news_url, params=news_params) as response:
                if response.status == 200:
                    news_data = await response.json()
                    for article in news_data[:limit//2]:
                        data.append({
                            'type': 'financial_news',
                            'title': article.get('title', ''),
                            'content': article.get('text', ''),
                            'url': article.get('url', ''),
                            'published_date': article.get('publishedDate', ''),
                            'source': 'Financial Modeling Prep',
                            'raw_data': article
                        })
        
        except Exception as e:
            logger.error(f"FMP API error: {e}")
        
        return data
    
    async def _fetch_alpha_vantage_data(self, ticker: str, limit: int) -> List[Dict]:
        """Fetch data from Alpha Vantage"""
        session = await self._get_session()
        data = []
        
        try:
            # Company overview
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }
            
            async with session.get(self.alpha_vantage_base_url, params=params) as response:
                if response.status == 200:
                    overview_data = await response.json()
                    if 'Symbol' in overview_data:
                        data.append({
                            'type': 'company_overview',
                            'title': f'Company Overview - {overview_data.get("Name", ticker)}',
                            'content': overview_data.get('Description', ''),
                            'sector': overview_data.get('Sector', ''),
                            'industry': overview_data.get('Industry', ''),
                            'market_cap': overview_data.get('MarketCapitalization', 0),
                            'pe_ratio': overview_data.get('PERatio', 0),
                            'dividend_yield': overview_data.get('DividendYield', 0),
                            'source': 'Alpha Vantage',
                            'raw_data': overview_data
                        })
        
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
        
        return data
    
    async def _fetch_polygon_data(self, ticker: str, limit: int) -> List[Dict]:
        """Fetch data from Polygon.io"""
        session = await self._get_session()
        data = []
        
        try:
            # Company details
            url = f"{self.polygon_base_url}/v3/reference/tickers/{ticker}"
            params = {'apikey': self.polygon_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    ticker_data = await response.json()
                    if 'results' in ticker_data:
                        company = ticker_data['results']
                        data.append({
                            'type': 'ticker_details',
                            'title': f'Ticker Details - {company.get("name", ticker)}',
                            'content': company.get('description', ''),
                            'url': company.get('homepage_url', ''),
                            'market': company.get('market', ''),
                            'locale': company.get('locale', ''),
                            'source': 'Polygon.io',
                            'raw_data': company
                        })
        
        except Exception as e:
            logger.error(f"Polygon API error: {e}")
        
        return data
    
    def _generate_placeholder_data(self, ticker: str) -> List[Dict]:
        """Generate placeholder financial data when no APIs are configured"""
        return [{
            'type': 'financial_placeholder',
            'title': f'Financial data for {ticker}',
            'content': 'Financial API integration not yet configured. Please add API keys for Financial Modeling Prep, Alpha Vantage, or Polygon.io in the configuration.',
            'source': 'Financial Fetcher',
            'ticker': ticker,
            'published_date': datetime.now(timezone.utc).isoformat(),
            'raw_data': {'placeholder': True}
        }]
    
    async def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """Get detailed company profile"""
        try:
            data = await self.fetch_all(ticker, limit=1)
            for item in data:
                if item.get('type') in ['company_profile', 'company_overview', 'ticker_details']:
                    return item
            return None
        except Exception as e:
            logger.error(f"Error getting company profile: {e}")
            return None
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test financial API connections"""
        results = {
            'success': False,
            'message': 'No financial APIs configured',
            'fetcher': 'Financial',
            'details': {}
        }
        
        if not self.is_configured():
            return results
        
        session = await self._get_session()
        working_apis = []
        
        # Test FMP
        if self.fmp_key:
            try:
                url = f"{self.fmp_base_url}/profile/AAPL"
                params = {'apikey': self.fmp_key}
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        working_apis.append('Financial Modeling Prep')
                        results['details']['fmp'] = 'Connected'
                    else:
                        results['details']['fmp'] = f'Error {response.status}'
            except Exception as e:
                results['details']['fmp'] = f'Failed: {str(e)[:50]}'
        
        # Test Alpha Vantage
        if self.alpha_vantage_key:
            try:
                params = {
                    'function': 'OVERVIEW',
                    'symbol': 'AAPL',
                    'apikey': self.alpha_vantage_key
                }
                async with session.get(self.alpha_vantage_base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Symbol' in data:
                            working_apis.append('Alpha Vantage')
                            results['details']['alpha_vantage'] = 'Connected'
                        else:
                            results['details']['alpha_vantage'] = 'Invalid response'
                    else:
                        results['details']['alpha_vantage'] = f'Error {response.status}'
            except Exception as e:
                results['details']['alpha_vantage'] = f'Failed: {str(e)[:50]}'
        
        # Test Polygon
        if self.polygon_key:
            try:
                url = f"{self.polygon_base_url}/v3/reference/tickers/AAPL"
                params = {'apikey': self.polygon_key}
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        working_apis.append('Polygon.io')
                        results['details']['polygon'] = 'Connected'
                    else:
                        results['details']['polygon'] = f'Error {response.status}'
            except Exception as e:
                results['details']['polygon'] = f'Failed: {str(e)[:50]}'
        
        if working_apis:
            results['success'] = True
            results['message'] = f"Connected to: {', '.join(working_apis)}"
        else:
            results['message'] = "All financial API connections failed"
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().cleanup()
