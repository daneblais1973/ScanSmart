import json
import pickle
import logging
import time
import threading
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import hashlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import AppConfig

logger = logging.getLogger(__name__)

class MemoryCache:
    """In-memory cache implementation as fallback"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value with TTL"""
        if ttl is None:
            ttl = self.default_ttl
            
        expiry = time.time() + ttl
        
        with self._lock:
            # Clean up expired entries if cache is full
            if len(self._cache) >= self.max_size:
                self._cleanup_expired()
            
            # If still full, remove oldest entries
            if len(self._cache) >= self.max_size:
                sorted_keys = sorted(self._cache.keys(), 
                                   key=lambda k: self._cache[k]['created'])
                for old_key in sorted_keys[:100]:  # Remove 100 oldest
                    del self._cache[old_key]
            
            self._cache[key] = {
                'value': value,
                'expiry': expiry,
                'created': time.time()
            }
            
        return True
    
    def get(self, key: str) -> Any:
        """Get cache value"""
        with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            
            # Check expiry
            if time.time() > entry['expiry']:
                del self._cache[key]
                return None
                
            return entry['value']
    
    def delete(self, key: str) -> bool:
        """Delete cache key"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expiry']
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            current_time = time.time()
            expired_count = sum(
                1 for entry in self._cache.values()
                if current_time > entry['expiry']
            )
            
        return {
            'type': 'memory',
            'total_entries': total_entries,
            'expired_entries': expired_count,
            'active_entries': total_entries - expired_count,
            'max_size': self.max_size
        }

class RedisCache:
    """Redis-based cache implementation"""
    
    def __init__(self, redis_url: str, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._redis = redis.from_url(redis_url, decode_responses=False)
        
        # Test connection
        try:
            self._redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value with TTL"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Set with expiration
            result = self._redis.setex(key, ttl, serialized)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Any:
        """Get cache value"""
        try:
            data = self._redis.get(key)
            if data is None:
                return None
                
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache key"""
        try:
            result = self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries (flushdb)"""
        try:
            self._redis.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            info = self._redis.info()
            return {
                'type': 'redis',
                'total_keys': info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {'type': 'redis', 'error': str(e)}

class CacheManager:
    """Main cache manager that handles both Redis and memory cache"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        self._lock = threading.Lock()
        
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache backend"""
        if not self.config.cache.enabled:
            logger.info("Cache disabled in configuration")
            return
        
        # Try Redis first, fallback to memory cache
        if REDIS_AVAILABLE and self.config.cache.redis_url:
            try:
                self._cache = RedisCache(
                    self.config.cache.redis_url,
                    self.config.cache.default_ttl
                )
                logger.info("Using Redis cache")
                return
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        # Fallback to memory cache
        self._cache = MemoryCache(
            max_size=1000,
            default_ttl=self.config.cache.default_ttl
        )
        logger.info("Using memory cache")
    
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace"""
        return f"catalyst_scanner:{namespace}:{key}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> bool:
        """Set cache value"""
        if not self._cache:
            return False
        
        try:
            cache_key = self._generate_key(key, namespace)
            result = self._cache.set(cache_key, value, ttl)
            
            with self._lock:
                self._stats['sets'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            with self._lock:
                self._stats['errors'] += 1
            return False
    
    def get(self, key: str, namespace: str = "default") -> Any:
        """Get cache value"""
        if not self._cache:
            return None
        
        try:
            cache_key = self._generate_key(key, namespace)
            value = self._cache.get(cache_key)
            
            with self._lock:
                if value is not None:
                    self._stats['hits'] += 1
                else:
                    self._stats['misses'] += 1
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            with self._lock:
                self._stats['errors'] += 1
            return None
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete cache key"""
        if not self._cache:
            return False
        
        try:
            cache_key = self._generate_key(key, namespace)
            result = self._cache.delete(cache_key)
            
            with self._lock:
                self._stats['deletes'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            with self._lock:
                self._stats['errors'] += 1
            return False
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists"""
        if not self._cache:
            return False
        
        try:
            cache_key = self._generate_key(key, namespace)
            return self._cache.exists(cache_key)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def clear(self, namespace: Optional[str] = None):
        """Clear cache entries"""
        if not self._cache:
            return
        
        try:
            if namespace:
                # Clear specific namespace (Redis only feature)
                if hasattr(self._cache, '_redis'):
                    pattern = f"catalyst_scanner:{namespace}:*"
                    keys = self._cache._redis.keys(pattern)
                    if keys:
                        self._cache._redis.delete(*keys)
                else:
                    # Memory cache doesn't support pattern deletion
                    logger.warning("Namespace clearing not supported for memory cache")
            else:
                self._cache.clear()
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_ops = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_ops if total_ops > 0 else 0
            
            base_stats = {
                'hit_rate': hit_rate,
                'total_operations': total_ops,
                **self._stats
            }
        
        # Add backend-specific stats
        if self._cache:
            try:
                backend_stats = self._cache.get_stats()
                base_stats.update(backend_stats)
            except Exception as e:
                base_stats['backend_error'] = str(e)
        
        return base_stats
    
    def cache_decorator(self, key_func=None, ttl=None, namespace="default"):
        """Decorator for caching function results"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__] + [str(arg) for arg in args]
                    if kwargs:
                        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                    cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
                
                # Try cache first
                cached_result = self.get(cache_key, namespace)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl, namespace)
                
                return result
            
            return wrapper
        return decorator
