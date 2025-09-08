import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from core.config import AppConfig
from core.database import DatabaseManager
from shared.models import Catalyst, SentimentLabel, CatalystType

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Results from backtesting analysis"""
    test_period_start: datetime
    test_period_end: datetime
    total_catalysts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    avg_detection_time: float
    category_performance: Dict[str, Dict[str, float]]

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    lookback_days: int = 365
    validation_threshold: float = 0.7
    min_impact_score: int = 50
    time_window_hours: int = 24
    benchmark_sources: List[str] = None

class HistoricalValidator:
    """Backtesting framework for catalyst detection accuracy validation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.backtest_config = BacktestConfig()
        
        self.results_dir = Path('backtesting/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.historical_data = []
        self.benchmark_data = []
        
        logger.info("Historical Validator initialized")
    
    async def load_historical_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load historical catalyst data for backtesting"""
        try:
            # In a real implementation, this would query historical database
            # For demonstration, we'll create representative historical data
            
            historical_catalysts = []
            current_date = start_date
            
            # Sample historical catalyst patterns
            catalyst_templates = [
                {
                    "ticker": "AAPL",
                    "catalyst": "Apple reports strong quarterly earnings",
                    "category": "Earnings / Guidance",
                    "sentiment_label": "positive",
                    "impact": 85,
                    "actual_price_change": 0.045  # 4.5% increase
                },
                {
                    "ticker": "TSLA", 
                    "catalyst": "Tesla announces new factory expansion",
                    "category": "M&A / Partnerships",
                    "sentiment_label": "positive",
                    "impact": 75,
                    "actual_price_change": 0.032
                },
                {
                    "ticker": "GOOGL",
                    "catalyst": "Google faces antitrust investigation",
                    "category": "Regulatory / FDA",
                    "sentiment_label": "negative", 
                    "impact": 70,
                    "actual_price_change": -0.028
                },
                {
                    "ticker": "MSFT",
                    "catalyst": "Microsoft Azure revenue grows 50%",
                    "category": "Earnings / Guidance",
                    "sentiment_label": "positive",
                    "impact": 80,
                    "actual_price_change": 0.035
                }
            ]
            
            # Generate historical data over the time period
            days_total = (end_date - start_date).days
            for day in range(days_total):
                current_date = start_date + timedelta(days=day)
                
                # Simulate 1-3 catalysts per day
                num_catalysts = np.random.poisson(1.5)
                
                for _ in range(min(num_catalysts, 5)):
                    template = np.random.choice(catalyst_templates)
                    
                    # Add some randomness to the template
                    catalyst = template.copy()
                    catalyst['published_date'] = current_date
                    catalyst['detected_time'] = current_date + timedelta(minutes=np.random.randint(5, 120))
                    catalyst['confidence'] = np.random.uniform(0.6, 0.95)
                    
                    # Add some noise to price changes
                    catalyst['actual_price_change'] *= (1 + np.random.uniform(-0.3, 0.3))
                    
                    historical_catalysts.append(catalyst)
            
            logger.info(f"Loaded {len(historical_catalysts)} historical catalyst records")
            return historical_catalysts
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return []
    
    async def load_benchmark_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load benchmark data for validation (actual market events)"""
        try:
            # In real implementation, this would load from financial data providers
            # For demo, create benchmark events that should have been detected
            
            benchmark_events = [
                {
                    "date": start_date + timedelta(days=30),
                    "ticker": "AAPL",
                    "event": "Apple Q1 earnings beat",
                    "category": "Earnings / Guidance",
                    "actual_impact": 8.5,  # Actual stock price change %
                    "should_detect": True
                },
                {
                    "date": start_date + timedelta(days=60),
                    "ticker": "TSLA", 
                    "event": "Tesla deliveries exceed guidance",
                    "category": "General News",
                    "actual_impact": 6.2,
                    "should_detect": True
                },
                {
                    "date": start_date + timedelta(days=90),
                    "ticker": "GOOGL",
                    "event": "Google announces AI partnership",
                    "category": "M&A / Partnerships", 
                    "actual_impact": 4.1,
                    "should_detect": True
                }
            ]
            
            logger.info(f"Loaded {len(benchmark_events)} benchmark events")
            return benchmark_events
            
        except Exception as e:
            logger.error(f"Failed to load benchmark data: {e}")
            return []
    
    def calculate_detection_accuracy(self, detected: List[Dict], benchmark: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics by comparing detected vs benchmark events"""
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Convert to sets for comparison
        detected_events = set()
        benchmark_events = set()
        
        for event in detected:
            key = (event['ticker'], event['published_date'].date(), event['category'])
            detected_events.add(key)
        
        for event in benchmark:
            if event['should_detect']:
                key = (event['ticker'], event['date'].date(), event['category'])
                benchmark_events.add(key)
        
        # Calculate metrics
        true_positives = len(detected_events.intersection(benchmark_events))
        false_positives = len(detected_events - benchmark_events)
        false_negatives = len(benchmark_events - detected_events)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(benchmark_events) if len(benchmark_events) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def analyze_category_performance(self, detected: List[Dict], benchmark: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by catalyst category"""
        categories = set([c['category'] for c in detected + benchmark])
        category_performance = {}
        
        for category in categories:
            cat_detected = [c for c in detected if c['category'] == category]
            cat_benchmark = [c for c in benchmark if c['category'] == category]
            
            if cat_detected or cat_benchmark:
                cat_metrics = self.calculate_detection_accuracy(cat_detected, cat_benchmark)
                category_performance[category] = cat_metrics
        
        return category_performance
    
    def calculate_timing_accuracy(self, detected: List[Dict]) -> float:
        """Calculate average detection time accuracy"""
        detection_times = []
        
        for catalyst in detected:
            if 'published_date' in catalyst and 'detected_time' in catalyst:
                detection_delay = (catalyst['detected_time'] - catalyst['published_date']).total_seconds() / 60
                detection_times.append(detection_delay)
        
        return np.mean(detection_times) if detection_times else 0.0
    
    async def run_backtest(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run comprehensive backtest analysis"""
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Load data
            historical_data = await self.load_historical_data(start_date, end_date)
            benchmark_data = await self.load_benchmark_data(start_date, end_date)
            
            # Filter by minimum impact score
            filtered_detected = [
                c for c in historical_data 
                if c.get('impact', 0) >= self.backtest_config.min_impact_score
            ]
            
            # Calculate accuracy metrics
            accuracy_metrics = self.calculate_detection_accuracy(filtered_detected, benchmark_data)
            
            # Analyze category performance
            category_performance = self.analyze_category_performance(filtered_detected, benchmark_data)
            
            # Calculate timing accuracy
            avg_detection_time = self.calculate_timing_accuracy(filtered_detected)
            
            # Create result
            result = BacktestResult(
                test_period_start=start_date,
                test_period_end=end_date,
                total_catalysts=len(filtered_detected),
                true_positives=accuracy_metrics['true_positives'],
                false_positives=accuracy_metrics['false_positives'],
                false_negatives=accuracy_metrics['false_negatives'],
                precision=accuracy_metrics['precision'],
                recall=accuracy_metrics['recall'],
                f1_score=accuracy_metrics['f1_score'],
                accuracy=accuracy_metrics['accuracy'],
                avg_detection_time=avg_detection_time,
                category_performance=category_performance
            )
            
            # Save results
            await self.save_backtest_results(result)
            
            logger.info(f"Backtest completed - F1 Score: {result.f1_score:.3f}, Precision: {result.precision:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def save_backtest_results(self, result: BacktestResult):
        """Save backtest results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert result to dictionary
            result_dict = {
                'test_period_start': result.test_period_start.isoformat(),
                'test_period_end': result.test_period_end.isoformat(),
                'total_catalysts': result.total_catalysts,
                'true_positives': result.true_positives,
                'false_positives': result.false_positives,
                'false_negatives': result.false_negatives,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'accuracy': result.accuracy,
                'avg_detection_time': result.avg_detection_time,
                'category_performance': result.category_performance
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
    
    async def run_rolling_backtest(self, window_days: int = 30, step_days: int = 7) -> List[BacktestResult]:
        """Run rolling window backtests for trend analysis"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.backtest_config.lookback_days)
            
            results = []
            current_start = start_date
            
            while current_start + timedelta(days=window_days) <= end_date:
                window_end = current_start + timedelta(days=window_days)
                
                result = await self.run_backtest(current_start, window_end)
                results.append(result)
                
                current_start += timedelta(days=step_days)
            
            logger.info(f"Completed {len(results)} rolling backtests")
            return results
            
        except Exception as e:
            logger.error(f"Rolling backtest failed: {e}")
            return []
    
    def get_performance_summary(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Generate performance summary from backtest results"""
        if not results:
            return {}
        
        f1_scores = [r.f1_score for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        
        return {
            'total_backtests': len(results),
            'avg_f1_score': np.mean(f1_scores),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'f1_score_trend': 'improving' if f1_scores[-1] > f1_scores[0] else 'declining' if len(f1_scores) > 1 else 'stable',
            'best_f1_score': max(f1_scores),
            'worst_f1_score': min(f1_scores),
            'consistency_score': 1 - np.std(f1_scores)  # Higher is more consistent
        }