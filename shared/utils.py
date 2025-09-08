import json
import logging
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import aiohttp
import requests
from functools import wraps
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Utilities ---
def load_json_config(file_path: str, default: Dict = None) -> Dict:
    """Load configuration from JSON file"""
    if default is None:
        default = {}
    
    try:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading config from {file_path}: {e}")
    
    return default

def save_json_config(file_path: str, config: Dict) -> bool:
    """Save configuration to JSON file"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {file_path}: {e}")
        return False

def get_env_variable(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback"""
    import os
    return os.environ.get(key, default)

# --- API Utilities ---
async def async_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    timeout: int = 30
) -> Optional[Dict]:
    """Make asynchronous HTTP request"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HTTP error {response.status} for {url}")
                    return None
    except Exception as e:
        logger.error(f"Error making async request to {url}: {e}")
        return None

def sync_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    timeout: int = 30
) -> Optional[Dict]:
    """Make synchronous HTTP request"""
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"HTTP error {response.status_code} for {url}")
            return None
    except Exception as e:
        logger.error(f"Error making sync request to {url}: {e}")
        return None

# --- Time Utilities ---
def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return datetime.now()

def is_timestamp_recent(timestamp: datetime, max_age_seconds: int = 3600) -> bool:
    """Check if timestamp is recent"""
    return (datetime.now() - timestamp).total_seconds() <= max_age_seconds

# --- Data Processing Utilities ---
def filter_unique_items(items: List[Dict], key: str = "id") -> List[Dict]:
    """Filter unique items by key"""
    seen = set()
    unique_items = []
    for item in items:
        item_key = item.get(key)
        if item_key and item_key not in seen:
            seen.add(item_key)
            unique_items.append(item)
    return unique_items

def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary values"""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

# --- Text Processing Utilities ---
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?@$%&*()-]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'

def extract_tickers_from_text(text: str) -> List[str]:
    """Extract stock tickers from text"""
    # Match $TICKER or TICKER pattern
    ticker_pattern = r'\$([A-Z]{1,5})|(?<!\w)([A-Z]{1,5})(?!\w)'
    matches = re.findall(ticker_pattern, text.upper())
    
    tickers = set()
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and len(ticker) <= 5:  # Reasonable ticker length
            tickers.add(ticker)
    
    return list(tickers)

# --- Rate Limiting Utilities ---
class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        
        # Remove calls older than period
        self.calls = [call for call in self.calls if now - call < self.period]
        
        if len(self.calls) >= self.max_calls:
            # Calculate sleep time
            sleep_time = self.period - (now - self.calls[0])
            await asyncio.sleep(max(0, sleep_time))
            # Refresh calls list after sleep
            self.calls = [call for call in self.calls if now + sleep_time - call < self.period]
        
        self.calls.append(time.time())

def rate_limited(max_calls: int, period: float = 1.0):
    """Decorator for rate-limiting function calls"""
    limiter = RateLimiter(max_calls, period)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# --- Error Handling Utilities ---
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# --- Validation Utilities ---
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    # Simple validation for common formats
    pattern = r'^\+?[1-9]\d{1,14}$'
    clean_phone = re.sub(r'[^\d+]', '', phone)
    return bool(re.match(pattern, clean_phone))

# --- Serialization Utilities ---
def json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, 'dict'):
        return obj.dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, default=json_serializer, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error: {e}")
        return "{}"

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parsing error: {e}")
        return default

# --- Service Discovery Utilities ---
class ServiceDiscovery:
    """Simple service discovery"""
    def __init__(self):
        self.services = {}
    
    def register_service(self, name: str, url: str, metadata: Dict = None):
        """Register a service"""
        self.services[name] = {
            'url': url,
            'metadata': metadata or {},
            'last_seen': datetime.now()
        }
    
    def get_service_url(self, name: str) -> Optional[str]:
        """Get service URL"""
        service = self.services.get(name)
        return service['url'] if service else None
    
    def get_all_services(self) -> Dict:
        """Get all registered services"""
        return self.services

# --- Caching Utilities ---
def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def ttl_cache(ttl_seconds: int = 300):
    """Simple TTL cache decorator"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Check cache
            if cache_key in cache:
                value, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return value
                else:
                    del cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # Clean up old entries
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in cache.items()
                if current_time - timestamp >= ttl_seconds
            ]
            for key in expired_keys:
                del cache[key]
            
            return result
        
        return wrapper
    return decorator

# --- Financial Utilities ---
def format_market_cap(market_cap: float) -> str:
    """Format market cap in human readable form"""
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.1f}T"
    elif market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.1f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.1f}M"
    else:
        return f"${market_cap:,.0f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage with proper sign and decimals"""
    return f"{value:+.{decimals}f}%"

def calculate_sentiment_color(sentiment_score: float) -> str:
    """Get color for sentiment score"""
    if sentiment_score > 0.1:
        return "green"
    elif sentiment_score < -0.1:
        return "red"
    else:
        return "gray"

# Create global service discovery instance
service_discovery = ServiceDiscovery()
