import logging
import asyncio
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import threading
import time

from core.config import AppConfig
from core.database import DatabaseManager
from shared.models import Catalyst, SentimentLabel

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model evaluation"""
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: datetime
    validation_samples: int
    training_samples: int

@dataclass
class RetrainingConfig:
    """Configuration for model retraining"""
    min_new_samples: int = 1000
    performance_threshold: float = 0.75
    retraining_interval_hours: int = 168  # Weekly
    validation_split: float = 0.2
    max_training_samples: int = 50000

class ModelTrainer:
    """Automated ML model retraining system with performance monitoring"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.retraining_config = RetrainingConfig()
        
        self.models_dir = Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_history = []
        self.last_training = {}
        self._training_lock = threading.Lock()
        
        logger.info("Model Trainer initialized")
    
    async def evaluate_model_performance(self, model_name: str, test_data: List[Dict]) -> ModelPerformanceMetrics:
        """Evaluate current model performance on test data"""
        try:
            # Simulate model evaluation with rule-based scoring
            correct_predictions = 0
            total_predictions = len(test_data)
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for sample in test_data:
                actual_sentiment = sample.get('sentiment_label', 'neutral')
                predicted_sentiment = await self._predict_sentiment(sample['text'])
                
                if actual_sentiment == predicted_sentiment:
                    correct_predictions += 1
                
                # Calculate precision/recall metrics
                if predicted_sentiment == 'positive':
                    if actual_sentiment == 'positive':
                        true_positives += 1
                    else:
                        false_positives += 1
                elif actual_sentiment == 'positive':
                    false_negatives += 1
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return ModelPerformanceMetrics(
                model_name=model_name,
                version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                training_date=datetime.now(timezone.utc),
                validation_samples=total_predictions,
                training_samples=len(test_data)
            )
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            raise
    
    async def _predict_sentiment(self, text: str) -> str:
        """Simple sentiment prediction for evaluation purposes"""
        positive_terms = ['bullish', 'optimistic', 'growth', 'profit', 'gain', 'surge', 'strong', 'beat', 'positive']
        negative_terms = ['bearish', 'pessimistic', 'loss', 'decline', 'drop', 'weak', 'miss', 'negative', 'warning']
        
        text_lower = text.lower()
        pos_count = sum(1 for term in positive_terms if term in text_lower)
        neg_count = sum(1 for term in negative_terms if term in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    async def collect_training_data(self, days_back: int = 30) -> List[Dict]:
        """Collect recent catalyst data for training"""
        try:
            # Get recent catalysts from database
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            
            # Simulated training data collection
            training_samples = []
            
            # In a real implementation, this would query the database
            # For now, we'll create representative training data
            sample_catalysts = [
                {"text": "Apple reports record quarterly earnings beating analyst expectations", "sentiment_label": "positive"},
                {"text": "Tesla stock surges on strong delivery numbers", "sentiment_label": "positive"},
                {"text": "Microsoft announces major partnership deal", "sentiment_label": "positive"},
                {"text": "Amazon faces regulatory investigation concerns", "sentiment_label": "negative"},
                {"text": "Google stock declines on privacy lawsuit news", "sentiment_label": "negative"},
                {"text": "Meta reports mixed quarterly results", "sentiment_label": "neutral"}
            ]
            
            # Extend with variations
            for _ in range(min(self.retraining_config.max_training_samples // len(sample_catalysts), 100)):
                training_samples.extend(sample_catalysts)
            
            logger.info(f"Collected {len(training_samples)} training samples")
            return training_samples
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            return []
    
    async def should_retrain_model(self, model_name: str) -> Tuple[bool, str]:
        """Determine if model should be retrained based on performance and time"""
        try:
            # Check if enough time has passed
            last_training_time = self.last_training.get(model_name)
            if last_training_time:
                hours_since_training = (datetime.now(timezone.utc) - last_training_time).total_seconds() / 3600
                if hours_since_training < self.retraining_config.retraining_interval_hours:
                    return False, f"Only {hours_since_training:.1f} hours since last training"
            
            # Check if we have enough new data
            training_data = await self.collect_training_data()
            if len(training_data) < self.retraining_config.min_new_samples:
                return False, f"Insufficient training data: {len(training_data)} < {self.retraining_config.min_new_samples}"
            
            # Check current model performance
            test_data = training_data[-200:]  # Use last 200 samples for testing
            current_metrics = await self.evaluate_model_performance(model_name, test_data)
            
            if current_metrics.f1_score < self.retraining_config.performance_threshold:
                return True, f"Performance below threshold: {current_metrics.f1_score:.3f} < {self.retraining_config.performance_threshold}"
            
            return True, "Scheduled retraining due"
            
        except Exception as e:
            logger.error(f"Error checking retraining conditions: {e}")
            return False, f"Error: {str(e)}"
    
    async def train_model(self, model_name: str, training_data: List[Dict]) -> ModelPerformanceMetrics:
        """Train/retrain a model with new data"""
        try:
            with self._training_lock:
                logger.info(f"Starting training for {model_name} with {len(training_data)} samples")
                
                # Split data
                split_idx = int(len(training_data) * (1 - self.retraining_config.validation_split))
                train_data = training_data[:split_idx]
                val_data = training_data[split_idx:]
                
                # Simulate training process
                await asyncio.sleep(2)  # Simulate training time
                
                # Evaluate on validation data
                metrics = await self.evaluate_model_performance(model_name, val_data)
                metrics.training_samples = len(train_data)
                
                # Save model (simulate)
                model_path = self.models_dir / f"{model_name}_{metrics.version}.pkl"
                with open(model_path, 'wb') as f:
                    # In real implementation, save actual trained model
                    model_data = {
                        'model_name': model_name,
                        'version': metrics.version,
                        'training_date': metrics.training_date.isoformat(),
                        'performance': metrics.__dict__
                    }
                    pickle.dump(model_data, f)
                
                # Update tracking
                self.last_training[model_name] = datetime.now(timezone.utc)
                self.performance_history.append(metrics)
                
                logger.info(f"Model {model_name} trained successfully - F1: {metrics.f1_score:.3f}")
                return metrics
                
        except Exception as e:
            logger.error(f"Model training failed for {model_name}: {e}")
            raise
    
    async def automated_retraining_cycle(self):
        """Run automated retraining cycle for all models"""
        models_to_train = ['sentiment_classifier', 'catalyst_detector', 'impact_scorer']
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'models_checked': len(models_to_train),
            'models_retrained': 0,
            'results': {}
        }
        
        for model_name in models_to_train:
            try:
                should_retrain, reason = await self.should_retrain_model(model_name)
                
                if should_retrain:
                    training_data = await self.collect_training_data()
                    if training_data:
                        metrics = await self.train_model(model_name, training_data)
                        results['models_retrained'] += 1
                        results['results'][model_name] = {
                            'status': 'retrained',
                            'metrics': metrics.__dict__,
                            'reason': reason
                        }
                    else:
                        results['results'][model_name] = {
                            'status': 'skipped',
                            'reason': 'No training data available'
                        }
                else:
                    results['results'][model_name] = {
                        'status': 'skipped',
                        'reason': reason
                    }
                    
            except Exception as e:
                results['results'][model_name] = {
                    'status': 'error',
                    'reason': str(e)
                }
                logger.error(f"Retraining failed for {model_name}: {e}")
        
        logger.info(f"Automated retraining cycle completed - {results['models_retrained']}/{results['models_checked']} models retrained")
        return results
    
    def get_performance_history(self, model_name: Optional[str] = None) -> List[Dict]:
        """Get historical performance metrics"""
        if model_name:
            return [m.__dict__ for m in self.performance_history if m.model_name == model_name]
        return [m.__dict__ for m in self.performance_history]
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        return {
            'last_training_times': {k: v.isoformat() for k, v in self.last_training.items()},
            'total_performance_records': len(self.performance_history),
            'available_models': list(self.models_dir.glob('*.pkl')),
            'retraining_config': self.retraining_config.__dict__
        }