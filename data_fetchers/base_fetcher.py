import asyncio
import logging
import os
import time
import threading
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FetcherError(Exception):
    """Custom exception for fetcher errors"""
    def __init__(self, message: str, error_code: str = "FETCH_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class BaseFetcher(ABC):
    """Base class for all data fetchers"""
    
    def __init__(self):
        self.fetcher_name = self.__class__.__name__
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_request_time': None,
            'avg_response_time': 0.0
        }
        
        # Configuration from environment or defaults
        source_name = self.get_source_name().upper()
        self.timeout = float(os.getenv(f"{source_name}_TIMEOUT", "20.0"))
        self.max_retries = int(os.getenv(f"{source_name}_MAX_RETRIES", "3"))
        self.cache_ttl = int(os.getenv(f"{source_name}_CACHE_TTL", "300"))
        self.enable_caching = os.getenv(f"{source_name}_ENABLE_CACHE", "true").lower() == "true"
        
        logger.info(f"{self.fetcher_name} initialized")
    
    @abstractmethod
    async def fetch_all(self, ticker: str, limit: int = 20, **kwargs) -> List[Dict]:
        """Fetch all data for a given ticker"""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the fetcher is properly configured"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the source name for this fetcher"""
        pass
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status"""
        return {
            'configured': self.is_configured(),
            'source_name': self.get_source_name(),
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'cache_enabled': self.enable_caching,
            'cache_ttl': self.cache_ttl
        }
    
    @staticmethod
    def validate_ticker(ticker: str) -> str:
        """Validate and normalize ticker symbol"""
        if not ticker or not isinstance(ticker, str):
            raise FetcherError("Ticker must be a non-empty string", "INVALID_TICKER")
        
        normalized = ticker.strip().upper()
        if not re.match(r'^[A-Z0-9.-]+$', normalized):
            raise FetcherError(f"Invalid ticker format: {ticker}", "INVALID_TICKER_FORMAT")
        
        if len(normalized) > 10:
            raise FetcherError(f"Ticker too long: {ticker}", "TICKER_TOO_LONG")
        
        return normalized
    
    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """Clean and sanitize text content"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if needed
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + '...'
        
        return cleaned
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        if not timestamp_str:
            return None
        
        try:
            # Handle ISO format
            if 'T' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            # Handle date-only format
            else:
                return datetime.strptime(timestamp_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse timestamp: {timestamp_str}")
            return None
    
    @staticmethod
    def current_timestamp() -> str:
        """Get current timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
    
    def generate_cache_key(self, method: str, **params) -> str:
        """Generate cache key for method with parameters"""
        sorted_params = sorted(params.items())
        params_str = json.dumps(sorted_params, default=str, sort_keys=True)
        key_content = f"{self.fetcher_name}:{method}:{params_str}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Get data from cache if available and not expired"""
        if not self.enable_caching:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self._stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                    return data
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
        
        self._stats['cache_misses'] += 1
        return None
    
    def set_cache(self, cache_key: str, data: List[Dict]):
        """Store data in cache"""
        if not self.enable_caching:
            return
        
        with self._cache_lock:
            # Prevent cache from growing too large
            if len(self._cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
                for old_key, _ in sorted_items[:100]:
                    del self._cache[old_key]
            
            self._cache[cache_key] = (data, time.time())
            logger.debug(f"Cached data for key: {cache_key[:16]}...")
    
    def normalize_data_item(self, raw_item: Dict, ticker: str) -> Dict:
        """Normalize data item to standard format"""
        return {
            'id': raw_item.get('id', hashlib.md5(str(raw_item).encode()).hexdigest()[:16]),
            'title': self.sanitize_text(raw_item.get('title', '')),
            'content': self.sanitize_text(raw_item.get('content', '')),
            'url': raw_item.get('url', ''),
            'published_date': self.parse_timestamp(raw_item.get('published_date', '')),
            'author': self.sanitize_text(raw_item.get('author', '')),
            'source': self.get_source_name(),
            'ticker': ticker,
            'raw_data': raw_item  # Keep original for debugging
        }
    
    def get_stats(self) -> Dict:
        """Get fetcher statistics"""
        with self._cache_lock:
            cache_size = len(self._cache)
        
        total_requests = self._stats['total_requests']
        success_rate = (self._stats['successful_requests'] / total_requests) if total_requests > 0 else 0
        
        return {
            'fetcher_name': self.fetcher_name,
            'source_name': self.get_source_name(),
            'is_configured': self.is_configured(),
            'success_rate': round(success_rate, 3),
            'cache_size': cache_size,
            'cache_hit_rate': self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses']) if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0 else 0,
            **self._stats
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        with self._cache_lock:
            self._cache.clear()
        logger.info(f"{self.fetcher_name} cache cleared")
    
    async def cleanup(self):
        """Cleanup resources (override in subclasses if needed)"""
        self.clear_cache()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test fetcher connection (implement in subclasses)"""
        return {
            'success': self.is_configured(),
            'message': 'Configuration check only',
            'fetcher': self.get_source_name()
        }
    
    def _update_stats(self, success: bool, response_time: Optional[float] = None):
        """Update internal statistics"""
        self._stats['total_requests'] += 1
        self._stats['last_request_time'] = datetime.now(timezone.utc).isoformat()
        
        if success:
            self._stats['successful_requests'] += 1
        else:
            self._stats['failed_requests'] += 1
        
        if response_time is not None:
            # Update rolling average response time
            if self._stats['avg_response_time'] == 0:
                self._stats['avg_response_time'] = response_time
            else:
                self._stats['avg_response_time'] = (self._stats['avg_response_time'] * 0.9 + response_time * 0.1)
