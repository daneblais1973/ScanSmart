import asyncio
import logging
import aiohttp
import websockets
import json
from typing import Dict, List, Optional, Callable
from datetime import datetime, timezone

from .base_fetcher import BaseFetcher, FetcherError

logger = logging.getLogger(__name__)

class BinanceFetcher(BaseFetcher):
    """Binance API fetcher - free public market data, 20-100 messages/sec"""
    
    def __init__(self):
        super().__init__()
        self.rest_url = "https://api.binance.com/api/v3"
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self._session = None
        self._session_lock = asyncio.Lock()
        self._ws_connection = None
        
        # Major crypto pairs for market analysis
        self.major_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT'
        ]
        
        logger.info("Binance Fetcher initialized - No API key required for public data")
    
    def get_source_name(self) -> str:
        return "Binance"
    
    def is_configured(self) -> bool:
        """Binance public data doesn't require configuration"""
        return True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {
                    'User-Agent': 'QuantumCatalyst-Binance/1.0 (Crypto Market Analysis)'
                }
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers=headers
                )
        return self._session
    
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch crypto market data"""
        try:
            # Get market data for major crypto pairs
            market_data = await self.get_market_data(limit)
            
            # Get 24hr ticker statistics
            ticker_stats = await self.get_ticker_stats(limit//2)
            
            all_data = market_data + ticker_stats
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error in Binance fetch_all: {e}")
            return []
    
    async def get_market_data(self, limit: int = 20) -> List[Dict]:
        """Get current market data for major crypto pairs"""
        try:
            # Check cache first
            cache_key = self.generate_cache_key("market_data", limit=limit)
            cached_result = self.get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            session = await self._get_session()
            all_data = []
            
            # Get ticker prices for all symbols
            url = f"{self.rest_url}/ticker/price"
            
            self._stats['total_requests'] += 1
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter for major pairs and process
                    for item in data:
                        symbol = item.get('symbol')
                        if symbol in self.major_pairs:
                            market_item = {
                                'type': 'crypto_price',
                                'symbol': symbol,
                                'price': float(item.get('price', 0)),
                                'source': 'Binance',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(market_item, symbol)
                            all_data.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                else:
                    logger.warning(f"Binance API returned status {response.status}")
                    self._stats['failed_requests'] += 1
            
            # Cache results
            self.set_cache(cache_key, all_data)
            
            logger.info(f"Binance: Fetched {len(all_data)} market prices")
            return all_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting Binance market data: {e}")
            return []
    
    async def get_ticker_stats(self, limit: int = 10) -> List[Dict]:
        """Get 24hr ticker statistics"""
        try:
            session = await self._get_session()
            url = f"{self.rest_url}/ticker/24hr"
            
            self._stats['total_requests'] += 1
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    stats_data = []
                    count = 0
                    
                    for item in data:
                        if count >= limit:
                            break
                            
                        symbol = item.get('symbol')
                        if symbol in self.major_pairs:
                            stats_item = {
                                'type': 'crypto_24h_stats',
                                'symbol': symbol,
                                'price_change': float(item.get('priceChange', 0)),
                                'price_change_percent': float(item.get('priceChangePercent', 0)),
                                'weighted_avg_price': float(item.get('weightedAvgPrice', 0)),
                                'prev_close_price': float(item.get('prevClosePrice', 0)),
                                'last_price': float(item.get('lastPrice', 0)),
                                'bid_price': float(item.get('bidPrice', 0)),
                                'ask_price': float(item.get('askPrice', 0)),
                                'open_price': float(item.get('openPrice', 0)),
                                'high_price': float(item.get('highPrice', 0)),
                                'low_price': float(item.get('lowPrice', 0)),
                                'volume': float(item.get('volume', 0)),
                                'quote_volume': float(item.get('quoteVolume', 0)),
                                'open_time': datetime.fromtimestamp(item.get('openTime', 0) / 1000, tz=timezone.utc),
                                'close_time': datetime.fromtimestamp(item.get('closeTime', 0) / 1000, tz=timezone.utc),
                                'count': int(item.get('count', 0)),
                                'source': 'Binance',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': item
                            }
                            
                            normalized = self.normalize_data_item(stats_item, symbol)
                            stats_data.append(normalized)
                            count += 1
                    
                    self._stats['successful_requests'] += 1
                    logger.info(f"Binance: Fetched {len(stats_data)} ticker statistics")
                    return stats_data
                    
                else:
                    logger.warning(f"Binance ticker stats returned status {response.status}")
                    self._stats['failed_requests'] += 1
                    return []
            
        except Exception as e:
            logger.error(f"Error getting Binance ticker stats: {e}")
            return []
    
    async def get_klines(self, symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 100) -> List[Dict]:
        """Get kline/candlestick data"""
        try:
            session = await self._get_session()
            url = f"{self.rest_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            self._stats['total_requests'] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    klines = []
                    for kline in data:
                        kline_item = {
                            'type': 'crypto_kline',
                            'symbol': symbol,
                            'open_time': datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                            'open_price': float(kline[1]),
                            'high_price': float(kline[2]),
                            'low_price': float(kline[3]),
                            'close_price': float(kline[4]),
                            'volume': float(kline[5]),
                            'close_time': datetime.fromtimestamp(kline[6] / 1000, tz=timezone.utc),
                            'quote_asset_volume': float(kline[7]),
                            'number_of_trades': int(kline[8]),
                            'taker_buy_base_asset_volume': float(kline[9]),
                            'taker_buy_quote_asset_volume': float(kline[10]),
                            'source': 'Binance',
                            'timestamp': datetime.now(timezone.utc),
                            'raw_data': kline
                        }
                        
                        normalized = self.normalize_data_item(kline_item, f"{symbol}_{interval}")
                        klines.append(normalized)
                    
                    self._stats['successful_requests'] += 1
                    return klines
                    
                else:
                    logger.warning(f"Binance klines returned status {response.status}")
                    self._stats['failed_requests'] += 1
                    return []
            
        except Exception as e:
            logger.error(f"Error getting Binance klines: {e}")
            return []
    
    async def start_websocket_stream(self, symbols: List[str] = None, callback: Callable = None):
        """Start WebSocket stream for real-time data"""
        if symbols is None:
            symbols = self.major_pairs[:5]  # Limit to top 5 to avoid overwhelming
        
        # Create stream names
        streams = []
        for symbol in symbols:
            streams.append(f"{symbol.lower()}@ticker")
        
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                self._ws_connection = websocket
                logger.info(f"Binance WebSocket connected for {len(symbols)} symbols")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if callback:
                            await callback(data)
                        
                        # Process the real-time data
                        if 'data' in data:
                            stream_data = data['data']
                            processed = {
                                'type': 'crypto_realtime',
                                'symbol': stream_data.get('s'),
                                'price': float(stream_data.get('c', 0)),
                                'price_change': float(stream_data.get('P', 0)),
                                'volume': float(stream_data.get('v', 0)),
                                'high': float(stream_data.get('h', 0)),
                                'low': float(stream_data.get('l', 0)),
                                'source': 'Binance_WS',
                                'timestamp': datetime.now(timezone.utc),
                                'raw_data': data
                            }
                            
                            logger.debug(f"Binance WS: {processed['symbol']} @ {processed['price']}")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Binance WebSocket JSON decode error: {e}")
                    except Exception as e:
                        logger.warning(f"Binance WebSocket processing error: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed")
        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}")
        finally:
            self._ws_connection = None
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test Binance API connection"""
        try:
            session = await self._get_session()
            
            # Test with server time endpoint (lightest request)
            url = f"{self.rest_url}/time"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'serverTime' in data:
                        return {
                            'success': True,
                            'message': "Binance API connection successful",
                            'fetcher': 'Binance'
                        }
            
            return {
                'success': False,
                'message': f"Binance API returned status {response.status}",
                'fetcher': 'Binance'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Binance connection failed: {str(e)}",
                'fetcher': 'Binance'
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._ws_connection:
            await self._ws_connection.close()
        
        if self._session and not self._session.closed:
            await self._session.close()
            
        await super().cleanup()