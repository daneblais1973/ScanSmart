import asyncio
import logging
import sys
from typing import Dict, List, Any
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMLVerificationSystem:
    """Comprehensive verification system for AI/ML analysis methods"""
    
    def __init__(self):
        self.test_results = {}
        self.verification_passed = True
        
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive verification of all AI/ML components"""
        try:
            logger.info("🔬 Starting comprehensive AI/ML verification...")
            
            # Test data for verification
            test_data = self._get_test_catalyst_data()
            
            # Verification tests
            tests = [
                ("NLP Processor Integration", self._verify_nlp_processor),
                ("Model Loading Verification", self._verify_model_loading),
                ("Sentiment Analysis Accuracy", self._verify_sentiment_analysis),
                ("Catalyst Detection Quality", self._verify_catalyst_detection),
                ("Data Validation Integration", self._verify_data_validation),
                ("Rule-Based Fallback System", self._verify_rule_based_fallback),
                ("Processing Performance", self._verify_processing_performance),
                ("Integration with Data Sources", self._verify_data_source_integration)
            ]
            
            # Run all verification tests
            for test_name, test_func in tests:
                try:
                    logger.info(f"Running test: {test_name}")
                    result = await test_func(test_data)
                    self.test_results[test_name] = result
                    
                    if not result.get('passed', False):
                        self.verification_passed = False
                        logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
                    else:
                        logger.info(f"✅ {test_name} PASSED")
                        
                except Exception as e:
                    self.verification_passed = False
                    self.test_results[test_name] = {
                        'passed': False,
                        'error': str(e),
                        'details': f"Test execution failed: {e}"
                    }
                    logger.error(f"❌ {test_name} CRASHED: {e}")
            
            # Generate final report
            final_report = self._generate_verification_report()
            
            if self.verification_passed:
                logger.info("🎉 All AI/ML verification tests PASSED!")
            else:
                logger.error("⚠️  Some AI/ML verification tests FAILED!")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Critical error in AI/ML verification: {e}")
            return {
                'overall_status': 'CRITICAL_ERROR',
                'error': str(e),
                'verification_passed': False
            }
    
    def _get_test_catalyst_data(self) -> List[Dict[str, Any]]:
        """Generate test catalyst data for verification"""
        return [
            {
                'ticker': 'AAPL',
                'title': 'Apple Reports Strong Q4 Earnings, Beats Revenue Expectations',
                'content': 'Apple Inc. announced quarterly earnings that exceeded analyst expectations, with revenue reaching $89.5 billion compared to estimates of $87.3 billion. The company reported earnings per share of $1.65, beating consensus estimates.',
                'source': 'NewsAPI',
                'published_date': datetime.now(timezone.utc).isoformat(),
                'expected_category': 'earnings',
                'expected_sentiment': 'positive',
                'expected_impact': 'high'
            },
            {
                'ticker': 'MSFT',
                'title': 'Microsoft Announces Acquisition of AI Startup for $2.1 Billion',
                'content': 'Microsoft Corporation has agreed to acquire an artificial intelligence startup in a deal valued at $2.1 billion. The acquisition will strengthen Microsoft\'s position in the AI market and is expected to close in Q2.',
                'source': 'Reuters',
                'published_date': datetime.now(timezone.utc).isoformat(),
                'expected_category': 'ma',
                'expected_sentiment': 'positive',
                'expected_impact': 'high'
            },
            {
                'ticker': 'GILD',
                'title': 'FDA Approves Gilead\'s New Cancer Drug After Phase III Trial Success',
                'content': 'The Food and Drug Administration has approved Gilead Sciences new cancer treatment following successful Phase III clinical trials. The drug showed significant improvement in patient outcomes with minimal side effects.',
                'source': 'FDA',
                'published_date': datetime.now(timezone.utc).isoformat(),
                'expected_category': 'regulatory',
                'expected_sentiment': 'positive',
                'expected_impact': 'very_high'
            },
            {
                'ticker': 'TSLA',
                'title': 'Tesla Stock Drops as Q3 Deliveries Miss Analyst Expectations',
                'content': 'Tesla reported Q3 vehicle deliveries of 435,000 units, falling short of analyst expectations of 455,000 units. The company cited supply chain challenges and production delays at its new facilities.',
                'source': 'Bloomberg',
                'published_date': datetime.now(timezone.utc).isoformat(),
                'expected_category': 'earnings',
                'expected_sentiment': 'negative',
                'expected_impact': 'medium'
            }
        ]
    
    async def _verify_nlp_processor(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify NLP processor integration and functionality"""
        try:
            from core.config import AppConfig
            from nlp.processor import NLPProcessor
            
            # Initialize with test config
            config = AppConfig()
            nlp_processor = NLPProcessor(config)
            
            # Test processing
            results = []
            for test_item in test_data[:2]:  # Test first 2 items
                try:
                    # Convert to NewsItem format
                    from shared.models import NewsItem, SourceType
                    news_item = NewsItem(
                        title=test_item['title'],
                        content=test_item['content'],
                        source=SourceType.NEWSAPI,
                        published_date=datetime.fromisoformat(test_item['published_date'].replace('Z', '+00:00')),
                        url="https://test.example.com"
                    )
                    
                    # Process news item
                    catalyst = await nlp_processor.process_news_item(news_item, {'ticker': test_item['ticker']})
                    
                    if catalyst:
                        results.append({
                            'ticker': catalyst.ticker,
                            'category': catalyst.category.value,
                            'sentiment': catalyst.sentiment_label.value,
                            'impact': catalyst.impact,
                            'confidence': catalyst.confidence
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to process test item: {e}")
            
            return {
                'passed': len(results) > 0,
                'processed_items': len(results),
                'expected_items': 2,
                'details': {
                    'results': results,
                    'processor_stats': getattr(nlp_processor, '_stats', {}),
                    'processing_mode': getattr(nlp_processor._stats, 'processing_mode', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'NLP processor verification failed'
            }
    
    async def _verify_model_loading(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify that AI/ML models are loading correctly"""
        try:
            from nlp.models import ModelManager
            from core.config import AppConfig
            
            config = AppConfig()
            model_manager = ModelManager(config.nlp)
            
            # Test model loading
            model_tests = {}
            
            # Test FinBERT loading
            try:
                finbert = await model_manager.load_finbert()
                model_tests['finbert'] = {
                    'loaded': finbert.get('loaded', False),
                    'device': finbert.get('device', 'unknown'),
                    'mock': finbert.get('mock', False),
                    'error': finbert.get('error')
                }
            except Exception as e:
                model_tests['finbert'] = {'loaded': False, 'error': str(e)}
            
            # Test BART loading
            try:
                bart = await model_manager.load_bart()
                model_tests['bart'] = {
                    'loaded': bart.get('loaded', False),
                    'device': bart.get('device', 'unknown'),
                    'error': bart.get('error')
                }
            except Exception as e:
                model_tests['bart'] = {'loaded': False, 'error': str(e)}
            
            # Test spaCy loading
            try:
                spacy_model = await model_manager.load_spacy()
                model_tests['spacy'] = {
                    'loaded': spacy_model.get('loaded', False),
                    'device': spacy_model.get('device', 'unknown'),
                    'error': spacy_model.get('error')
                }
            except Exception as e:
                model_tests['spacy'] = {'loaded': False, 'error': str(e)}
            
            # Check if at least one model is working
            working_models = [name for name, result in model_tests.items() if result.get('loaded', False)]
            
            return {
                'passed': len(working_models) > 0,
                'working_models': working_models,
                'total_models': len(model_tests),
                'details': model_tests
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Model loading verification failed'
            }
    
    async def _verify_sentiment_analysis(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify sentiment analysis accuracy"""
        try:
            from nlp.sentiment_analyzer import SentimentAnalyzer
            from core.config import AppConfig
            
            config = AppConfig()
            sentiment_analyzer = SentimentAnalyzer(config.nlp)
            
            # Test sentiment analysis
            correct_predictions = 0
            total_predictions = 0
            results = []
            
            for test_item in test_data:
                try:
                    text = f"{test_item['title']} {test_item['content']}"
                    sentiment_result = await sentiment_analyzer.analyze_sentiment(text)
                    
                    predicted = sentiment_result['label'].value.lower()
                    expected = test_item['expected_sentiment'].lower()
                    
                    is_correct = predicted == expected
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    results.append({
                        'ticker': test_item['ticker'],
                        'predicted': predicted,
                        'expected': expected,
                        'correct': is_correct,
                        'confidence': sentiment_result.get('confidence', 0.0),
                        'score': sentiment_result.get('score', 0.0)
                    })
                    
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for {test_item['ticker']}: {e}")
            
            accuracy = correct_predictions / max(1, total_predictions)
            
            return {
                'passed': accuracy >= 0.5,  # At least 50% accuracy
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'details': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Sentiment analysis verification failed'
            }
    
    async def _verify_catalyst_detection(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify catalyst detection quality"""
        try:
            from nlp.rule_based_detector import rule_based_detector
            
            # Test rule-based detection
            correct_categories = 0
            total_tests = 0
            results = []
            
            for test_item in test_data:
                try:
                    text = f"{test_item['title']} {test_item['content']}"
                    detection_result = rule_based_detector.detect_catalysts(text, test_item['ticker'])
                    
                    if detection_result:
                        predicted_category = detection_result[0].get('category', 'unknown')
                        expected_category = test_item['expected_category']
                        
                        is_correct = predicted_category == expected_category
                        if is_correct:
                            correct_categories += 1
                        total_tests += 1
                        
                        results.append({
                            'ticker': test_item['ticker'],
                            'predicted_category': predicted_category,
                            'expected_category': expected_category,
                            'correct': is_correct,
                            'impact_score': detection_result[0].get('impact', 0),
                            'confidence': detection_result[0].get('confidence', 0.0)
                        })
                    else:
                        total_tests += 1
                        results.append({
                            'ticker': test_item['ticker'],
                            'predicted_category': 'none_detected',
                            'expected_category': test_item['expected_category'],
                            'correct': False,
                            'impact_score': 0,
                            'confidence': 0.0
                        })
                        
                except Exception as e:
                    logger.warning(f"Catalyst detection failed for {test_item['ticker']}: {e}")
            
            accuracy = correct_categories / max(1, total_tests)
            
            return {
                'passed': accuracy >= 0.4,  # At least 40% accuracy for rule-based
                'accuracy': accuracy,
                'correct_detections': correct_categories,
                'total_tests': total_tests,
                'details': results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Catalyst detection verification failed'
            }
    
    async def _verify_data_validation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify data validation integration"""
        try:
            from core.data_validator import DataValidator
            
            validator = DataValidator()
            
            # Test data validation
            validation_results = []
            high_quality_count = 0
            
            for test_item in test_data:
                try:
                    validation_result = validator.validate_catalyst_data(test_item)
                    
                    if validation_result['quality_score'] >= 0.7:
                        high_quality_count += 1
                    
                    validation_results.append({
                        'ticker': test_item['ticker'],
                        'is_valid': validation_result['is_valid'],
                        'quality_score': validation_result['quality_score'],
                        'confidence_score': validation_result['confidence_score'],
                        'risk_flags': len(validation_result['risk_flags'])
                    })
                    
                except Exception as e:
                    logger.warning(f"Data validation failed for {test_item['ticker']}: {e}")
            
            # Test filtering
            filtered_data = validator.filter_high_quality_data(test_data, min_quality=0.6)
            
            return {
                'passed': len(validation_results) > 0,
                'high_quality_rate': high_quality_count / max(1, len(validation_results)),
                'filtered_items': len(filtered_data),
                'original_items': len(test_data),
                'details': validation_results
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Data validation verification failed'
            }
    
    async def _verify_rule_based_fallback(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify rule-based fallback system"""
        try:
            from nlp.rule_based_detector import RuleBasedCatalystDetector
            
            detector = RuleBasedCatalystDetector()
            
            # Test rule-based detection
            detection_count = 0
            pattern_matches = 0
            
            for test_item in test_data:
                try:
                    text = f"{test_item['title']} {test_item['content']}"
                    catalysts = detector.detect_catalysts(text, test_item['ticker'])
                    
                    if catalysts:
                        detection_count += 1
                        # Check if patterns are working
                        if catalysts[0].get('confidence', 0) > 0.3:
                            pattern_matches += 1
                            
                except Exception as e:
                    logger.warning(f"Rule-based detection failed for {test_item['ticker']}: {e}")
            
            return {
                'passed': detection_count > 0,
                'detections': detection_count,
                'pattern_matches': pattern_matches,
                'total_tests': len(test_data),
                'detection_rate': detection_count / max(1, len(test_data))
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Rule-based fallback verification failed'
            }
    
    async def _verify_processing_performance(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify processing performance"""
        try:
            import time
            from nlp.processor import NLPProcessor
            from core.config import AppConfig
            
            config = AppConfig()
            nlp_processor = NLPProcessor(config)
            
            # Measure processing time
            start_time = time.time()
            processed_count = 0
            
            for test_item in test_data:
                try:
                    from shared.models import NewsItem, SourceType
                    news_item = NewsItem(
                        title=test_item['title'],
                        content=test_item['content'],
                        source=SourceType.NEWSAPI,
                        published_date=datetime.fromisoformat(test_item['published_date'].replace('Z', '+00:00')),
                        url="https://test.example.com"
                    )
                    
                    catalyst = await nlp_processor.process_news_item(news_item, {'ticker': test_item['ticker']})
                    if catalyst:
                        processed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Performance test failed for {test_item['ticker']}: {e}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            items_per_second = processed_count / max(0.1, processing_time)
            
            return {
                'passed': items_per_second > 0.5,  # At least 0.5 items per second
                'processing_time': processing_time,
                'processed_count': processed_count,
                'items_per_second': items_per_second,
                'performance_acceptable': items_per_second > 1.0
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Performance verification failed'
            }
    
    async def _verify_data_source_integration(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Verify integration with data sources"""
        try:
            from data_fetchers import get_all_fetchers
            from core.config import AppConfig
            
            config = AppConfig()
            fetchers = get_all_fetchers(config)
            
            # Test fetcher availability
            working_fetchers = []
            total_fetchers = len(fetchers)
            
            for name, fetcher in fetchers.items():
                try:
                    if hasattr(fetcher, 'is_configured') and fetcher.is_configured():
                        working_fetchers.append(name)
                    elif hasattr(fetcher, 'test_connection'):
                        test_result = await fetcher.test_connection()
                        if test_result.get('success', False):
                            working_fetchers.append(name)
                except Exception as e:
                    logger.debug(f"Fetcher {name} not available: {e}")
            
            return {
                'passed': len(working_fetchers) >= 3,  # At least 3 working fetchers
                'working_fetchers': working_fetchers,
                'total_fetchers': total_fetchers,
                'availability_rate': len(working_fetchers) / max(1, total_fetchers)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Data source integration verification failed'
            }
    
    def _generate_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        total_tests = len(self.test_results)
        
        return {
            'overall_status': 'PASSED' if self.verification_passed else 'FAILED',
            'verification_passed': self.verification_passed,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / max(1, total_tests),
            'test_results': self.test_results,
            'summary': {
                'ai_ml_models_working': any('model' in name.lower() for name, result in self.test_results.items() if result.get('passed', False)),
                'nlp_processor_working': self.test_results.get('NLP Processor Integration', {}).get('passed', False),
                'data_validation_working': self.test_results.get('Data Validation Integration', {}).get('passed', False),
                'rule_based_fallback_working': self.test_results.get('Rule-Based Fallback System', {}).get('passed', False),
                'performance_acceptable': self.test_results.get('Processing Performance', {}).get('passed', False)
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results.get('Model Loading Verification', {}).get('passed', False):
            recommendations.append("Install PyTorch and Transformers for full AI/ML capabilities")
        
        if not self.test_results.get('Sentiment Analysis Accuracy', {}).get('passed', False):
            recommendations.append("Review sentiment analysis model configuration")
        
        if not self.test_results.get('Processing Performance', {}).get('passed', False):
            recommendations.append("Optimize processing pipeline for better performance")
        
        if not self.test_results.get('Data Validation Integration', {}).get('passed', False):
            recommendations.append("Enhance data validation rules for better quality control")
        
        return recommendations

# Main execution function
async def main():
    """Run AI/ML verification system"""
    verifier = AIMLVerificationSystem()
    report = await verifier.run_comprehensive_verification()
    
    print("\n" + "="*80)
    print("🔬 AI/ML VERIFICATION REPORT")
    print("="*80)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    
    if report.get('recommendations'):
        print("\n📋 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())