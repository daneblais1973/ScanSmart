import logging
import os
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

# Optional ML dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        BartForConditionalGeneration, BartTokenizer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    from spacy.cli import download
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

from core.config import NLPConfig
from .rule_based_detector import rule_based_detector

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of NLP models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self._models = {}
        self._tokenizers = {}
        self._pipelines = {}
        self._load_lock = threading.Lock()
        
        # Ensure model cache directory exists
        Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine device
        self.device = self._determine_device()
        
        logger.info(f"ModelManager initialized - Device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine the best device for model inference"""
        if not TORCH_AVAILABLE:
            return 'cpu'  # Fallback when torch is not available
            
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        return self.config.device
    
    async def load_finbert(self) -> Dict[str, Any]:
        """Load FinBERT model for financial sentiment analysis"""
        model_name = 'finbert'
        
        if model_name in self._models:
            return self._models[model_name]
        
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("PyTorch or Transformers not available - using mock sentiment analysis")
            self._models[model_name] = {
                'model': None,
                'tokenizer': None,
                'pipeline': None,
                'device': 'cpu',
                'loaded': False,
                'mock': True,
                'error': 'ML dependencies not available'
            }
            return self._models[model_name]
        
        try:
            with self._load_lock:
                if model_name not in self._models:
                    logger.info(f"Loading FinBERT model: {self.config.finbert_model}")
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.finbert_model,
                        cache_dir=self.config.model_cache_dir
                    )
                    
                    # Load model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.finbert_model,
                        cache_dir=self.config.model_cache_dir
                    )
                    
                    # Move to device
                    model = model.to(self.device)
                    model.eval()
                    
                    # Create pipeline for easier use
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if self.device == 'cuda' else -1
                    )
                    
                    self._models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'pipeline': sentiment_pipeline,
                        'loaded': True,
                        'device': self.device
                    }
                    
                    logger.info("FinBERT model loaded successfully")
            
            return self._models[model_name]
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            self._models[model_name] = {'loaded': False, 'error': str(e)}
            return self._models[model_name]
    
    async def load_bart(self) -> Dict[str, Any]:
        """Load BART model for text summarization"""
        model_name = 'bart'
        
        if model_name in self._models:
            return self._models[model_name]
        
        try:
            with self._load_lock:
                if model_name not in self._models:
                    logger.info(f"Loading BART model: {self.config.bart_model}")
                    
                    # Load tokenizer
                    tokenizer = BartTokenizer.from_pretrained(
                        self.config.bart_model,
                        cache_dir=self.config.model_cache_dir
                    )
                    
                    # Load model
                    model = BartForConditionalGeneration.from_pretrained(
                        self.config.bart_model,
                        cache_dir=self.config.model_cache_dir
                    )
                    
                    # Move to device
                    model = model.to(self.device)
                    model.eval()
                    
                    # Create summarization pipeline
                    summarizer = pipeline(
                        "summarization",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if self.device == 'cuda' else -1
                    )
                    
                    self._models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'pipeline': summarizer,
                        'loaded': True,
                        'device': self.device
                    }
                    
                    logger.info("BART model loaded successfully")
            
            return self._models[model_name]
            
        except Exception as e:
            logger.error(f"Error loading BART model: {e}")
            self._models[model_name] = {'loaded': False, 'error': str(e)}
            return self._models[model_name]
    
    async def load_spacy(self) -> Dict[str, Any]:
        """Load spaCy model for NER and text processing"""
        model_name = 'spacy'
        
        if model_name in self._models:
            return self._models[model_name]
        
        try:
            with self._load_lock:
                if model_name not in self._models:
                    logger.info(f"Loading spaCy model: {self.config.spacy_model}")
                    
                    # Try to load the model
                    try:
                        nlp = spacy.load(self.config.spacy_model)
                    except OSError:
                        # Model not found, try to download it
                        logger.info(f"spaCy model not found, downloading: {self.config.spacy_model}")
                        download(self.config.spacy_model)
                        nlp = spacy.load(self.config.spacy_model)
                    
                    self._models[model_name] = {
                        'model': nlp,
                        'loaded': True,
                        'device': 'cpu'  # spaCy typically runs on CPU
                    }
                    
                    logger.info("spaCy model loaded successfully")
            
            return self._models[model_name]
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self._models[model_name] = {'loaded': False, 'error': str(e)}
            return self._models[model_name]
    
    async def get_finbert_pipeline(self):
        """Get FinBERT sentiment analysis pipeline"""
        finbert_data = await self.load_finbert()
        if finbert_data.get('loaded'):
            return finbert_data['pipeline']
        return None
    
    async def get_bart_pipeline(self):
        """Get BART summarization pipeline"""
        bart_data = await self.load_bart()
        if bart_data.get('loaded'):
            return bart_data['pipeline']
        return None
    
    async def get_spacy_model(self):
        """Get spaCy NLP model"""
        spacy_data = await self.load_spacy()
        if spacy_data.get('loaded'):
            return spacy_data['model']
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'loaded': False,
            'device': self.device,
            'models': {}
        }
        
        model_names = ['finbert', 'bart', 'spacy']
        loaded_count = 0
        
        for model_name in model_names:
            if model_name in self._models:
                model_data = self._models[model_name]
                status['models'][model_name] = {
                    'loaded': model_data.get('loaded', False),
                    'error': model_data.get('error'),
                    'device': model_data.get('device', 'unknown')
                }
                if model_data.get('loaded'):
                    loaded_count += 1
            else:
                status['models'][model_name] = {
                    'loaded': False,
                    'error': None,
                    'device': None
                }
        
        status['loaded'] = loaded_count > 0
        status['loaded_count'] = loaded_count
        status['total_models'] = len(model_names)
        
        return status
    
    async def preload_all_models(self):
        """Preload all models for faster inference"""
        try:
            logger.info("Preloading all NLP models...")
            
            # Load models concurrently
            tasks = [
                self.load_finbert(),
                self.load_bart(),
                self.load_spacy()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            loaded_count = 0
            for i, result in enumerate(results):
                model_name = ['finbert', 'bart', 'spacy'][i]
                if isinstance(result, dict) and result.get('loaded'):
                    loaded_count += 1
                    logger.info(f"✅ {model_name} model loaded")
                else:
                    error = result if isinstance(result, Exception) else result.get('error', 'Unknown error')
                    logger.error(f"❌ {model_name} model failed to load: {error}")
            
            logger.info(f"Model preloading completed: {loaded_count}/3 models loaded")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error during model preloading: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup model resources"""
        try:
            with self._load_lock:
                for model_name, model_data in self._models.items():
                    if model_data.get('loaded') and 'model' in model_data:
                        try:
                            # Clear model from GPU memory if applicable
                            if hasattr(model_data['model'], 'to'):
                                model_data['model'].to('cpu')
                            del model_data['model']
                            logger.debug(f"Cleaned up {model_name} model")
                        except Exception as e:
                            logger.warning(f"Error cleaning up {model_name}: {e}")
                
                self._models.clear()
                self._tokenizers.clear()
                self._pipelines.clear()
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
