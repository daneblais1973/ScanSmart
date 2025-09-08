# NLP module for financial catalyst detection and sentiment analysis
# Import NLP components with fallback handling
try:
    from .processor import NLPProcessor
    from .sentiment_analyzer import SentimentAnalyzer
    from .models import ModelManager
except ImportError:
    # If main components fail, use simple fallbacks
    from .simple_models import SimpleNLPProcessor as NLPProcessor
    from .simple_models import SimpleSentimentAnalyzer as SentimentAnalyzer
    from .simple_models import SimpleModelManager as ModelManager

__all__ = ['NLPProcessor', 'SentimentAnalyzer', 'ModelManager']
