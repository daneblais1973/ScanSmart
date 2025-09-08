import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """Professional-grade data validation to prevent false signals in stock screening"""
    
    def __init__(self):
        # Validation rules for professional stock screening
        self.min_confidence_threshold = 0.5
        self.max_age_hours = 72  # Maximum age for relevant data
        self.required_fields = ['ticker', 'title', 'source', 'published_date']
        
        # Known reliable sources (whitelist)
        self.trusted_sources = {
            'NewsAPI', 'Reuters', 'Bloomberg', 'Yahoo Finance', 'MarketWatch',
            'SEC EDGAR', 'FDA', 'Financial Times', 'Wall Street Journal',
            'AP News', 'Benzinga', 'TheStreet', 'Seeking Alpha'
        }
        
        # Suspicious keywords that indicate low-quality or speculative content
        self.suspicious_keywords = {
            'rumor', 'speculation', 'unconfirmed', 'alleged', 'supposedly',
            'may have', 'could be', 'might be', 'possibly', 'potentially',
            'sources say', 'insider claims', 'according to rumors'
        }
        
        # High-confidence keywords for professional validation
        self.high_confidence_keywords = {
            'announced', 'confirmed', 'official', 'filed', 'reported',
            'disclosed', 'released', 'issued', 'approved', 'completed'
        }
        
        logger.info("Professional Data Validator initialized")
    
    def validate_catalyst_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation for catalyst data to prevent false signals"""
        try:
            validation_result = {
                'is_valid': True,
                'confidence_score': 1.0,
                'validation_errors': [],
                'quality_score': 1.0,
                'risk_flags': []
            }
            
            # 1. Required field validation
            missing_fields = self._check_required_fields(data)
            if missing_fields:
                validation_result['is_valid'] = False
                validation_result['validation_errors'].append(f"Missing required fields: {missing_fields}")
                validation_result['confidence_score'] *= 0.3
            
            # 2. Data freshness validation
            age_penalty = self._validate_data_freshness(data)
            validation_result['confidence_score'] *= age_penalty
            
            # 3. Source credibility validation
            source_score = self._validate_source_credibility(data)
            validation_result['confidence_score'] *= source_score
            
            # 4. Content quality validation
            content_score, risk_flags = self._validate_content_quality(data)
            validation_result['confidence_score'] *= content_score
            validation_result['risk_flags'].extend(risk_flags)
            
            # 5. Ticker validation
            if not self._validate_ticker_format(data.get('ticker', '')):
                validation_result['validation_errors'].append("Invalid ticker format")
                validation_result['confidence_score'] *= 0.5
            
            # 6. Impact score validation
            impact_score = data.get('impact', 0)
            if not isinstance(impact_score, (int, float)) or impact_score < 0 or impact_score > 100:
                validation_result['validation_errors'].append("Invalid impact score")
                validation_result['confidence_score'] *= 0.7
            
            # 7. Professional threshold check
            if validation_result['confidence_score'] < self.min_confidence_threshold:
                validation_result['is_valid'] = False
                validation_result['validation_errors'].append(
                    f"Confidence score {validation_result['confidence_score']:.3f} below professional threshold {self.min_confidence_threshold}"
                )
            
            # 8. Calculate final quality score
            validation_result['quality_score'] = self._calculate_quality_score(data, validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating catalyst data: {e}")
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'validation_errors': [f"Validation error: {str(e)}"],
                'quality_score': 0.0,
                'risk_flags': ['validation_error']
            }
    
    def _check_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """Check for required fields"""
        missing = []
        for field in self.required_fields:
            if field not in data or not data[field]:
                missing.append(field)
        return missing
    
    def _validate_data_freshness(self, data: Dict[str, Any]) -> float:
        """Validate data freshness - recent data gets higher scores"""
        try:
            published_date = data.get('published_date')
            if not published_date:
                return 0.5  # No date = lower confidence
            
            # Parse date
            if isinstance(published_date, str):
                try:
                    pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                except:
                    pub_date = datetime.now(timezone.utc) - timedelta(days=30)  # Assume old
            else:
                pub_date = published_date
            
            # Calculate age in hours
            now = datetime.now(timezone.utc)
            age_hours = (now - pub_date).total_seconds() / 3600
            
            if age_hours < 0:
                return 0.1  # Future date = very suspicious
            elif age_hours <= 1:
                return 1.0  # Very fresh
            elif age_hours <= 6:
                return 0.95  # Fresh
            elif age_hours <= 24:
                return 0.9  # Recent
            elif age_hours <= self.max_age_hours:
                return 0.8  # Acceptable
            else:
                return 0.3  # Too old
                
        except Exception as e:
            logger.warning(f"Error validating data freshness: {e}")
            return 0.5
    
    def _validate_source_credibility(self, data: Dict[str, Any]) -> float:
        """Validate source credibility"""
        source = data.get('source', '').strip()
        if not source:
            return 0.3
        
        # Check against trusted sources
        for trusted in self.trusted_sources:
            if trusted.lower() in source.lower():
                return 1.0
        
        # Check for suspicious sources
        suspicious_indicators = ['blog', 'forum', 'social', 'unknown', 'anonymous']
        for indicator in suspicious_indicators:
            if indicator in source.lower():
                return 0.4
        
        # Default for unknown but not suspicious sources
        return 0.7
    
    def _validate_content_quality(self, data: Dict[str, Any]) -> tuple:
        """Validate content quality and detect risk flags"""
        content = f"{data.get('title', '')} {data.get('content', '')} {data.get('catalyst', '')}".lower()
        risk_flags = []
        quality_score = 1.0
        
        # Check for suspicious keywords
        suspicious_count = 0
        for keyword in self.suspicious_keywords:
            if keyword in content:
                suspicious_count += 1
                risk_flags.append(f"suspicious_keyword_{keyword}")
        
        if suspicious_count > 0:
            quality_score *= max(0.3, 1.0 - (suspicious_count * 0.2))
        
        # Check for high-confidence keywords
        high_conf_count = 0
        for keyword in self.high_confidence_keywords:
            if keyword in content:
                high_conf_count += 1
        
        if high_conf_count > 0:
            quality_score *= min(1.2, 1.0 + (high_conf_count * 0.1))
        
        # Check content length (too short might be low quality)
        if len(content.strip()) < 50:
            quality_score *= 0.6
            risk_flags.append("short_content")
        
        # Check for duplicate/repetitive content
        words = content.split()
        if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
            quality_score *= 0.7
            risk_flags.append("repetitive_content")
        
        return min(1.0, quality_score), risk_flags
    
    def _validate_ticker_format(self, ticker: str) -> bool:
        """Validate ticker symbol format"""
        if not ticker:
            return False
        
        # Basic ticker format: 1-5 uppercase letters
        ticker_pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(ticker_pattern, ticker.strip().upper()))
    
    def _calculate_quality_score(self, data: Dict[str, Any], validation_result: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        try:
            base_score = validation_result['confidence_score']
            
            # Adjust based on completeness
            completeness = len([f for f in self.required_fields if data.get(f)]) / len(self.required_fields)
            base_score *= completeness
            
            # Adjust based on risk flags
            risk_penalty = min(0.5, len(validation_result['risk_flags']) * 0.1)
            base_score *= (1.0 - risk_penalty)
            
            # Adjust based on numerical values
            if 'impact' in data and isinstance(data['impact'], (int, float)):
                if data['impact'] > 80:
                    base_score *= 1.1  # High impact gets boost
                elif data['impact'] < 30:
                    base_score *= 0.9  # Low impact gets penalty
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.5
    
    def filter_high_quality_data(self, data_list: List[Dict[str, Any]], 
                                min_quality: float = 0.7) -> List[Dict[str, Any]]:
        """Filter data list to only include high-quality, validated entries"""
        try:
            high_quality_data = []
            
            for item in data_list:
                validation_result = self.validate_catalyst_data(item)
                
                if (validation_result['is_valid'] and 
                    validation_result['quality_score'] >= min_quality):
                    
                    # Add validation metadata to the item
                    item['validation_metadata'] = {
                        'quality_score': validation_result['quality_score'],
                        'confidence_score': validation_result['confidence_score'],
                        'risk_flags': validation_result['risk_flags'],
                        'validated_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    high_quality_data.append(item)
                else:
                    logger.debug(f"Filtered out low-quality data: {item.get('title', 'Unknown')} "
                               f"(Quality: {validation_result['quality_score']:.3f})")
            
            logger.info(f"Filtered {len(data_list)} items to {len(high_quality_data)} high-quality entries")
            return high_quality_data
            
        except Exception as e:
            logger.error(f"Error filtering high-quality data: {e}")
            return []
    
    def get_validation_stats(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get validation statistics for a dataset"""
        try:
            stats = {
                'total_items': len(data_list),
                'valid_items': 0,
                'avg_quality_score': 0.0,
                'avg_confidence_score': 0.0,
                'common_risk_flags': {},
                'source_distribution': {},
                'validation_summary': {}
            }
            
            quality_scores = []
            confidence_scores = []
            all_risk_flags = []
            sources = []
            
            for item in data_list:
                validation_result = self.validate_catalyst_data(item)
                
                if validation_result['is_valid']:
                    stats['valid_items'] += 1
                
                quality_scores.append(validation_result['quality_score'])
                confidence_scores.append(validation_result['confidence_score'])
                all_risk_flags.extend(validation_result['risk_flags'])
                
                source = item.get('source', 'Unknown')
                sources.append(source)
            
            # Calculate averages
            if quality_scores:
                stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
            if confidence_scores:
                stats['avg_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
            
            # Count risk flags
            for flag in all_risk_flags:
                stats['common_risk_flags'][flag] = stats['common_risk_flags'].get(flag, 0) + 1
            
            # Count sources
            for source in sources:
                stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
            
            # Validation summary
            stats['validation_summary'] = {
                'pass_rate': stats['valid_items'] / max(1, stats['total_items']) * 100,
                'high_quality_rate': len([s for s in quality_scores if s >= 0.7]) / max(1, len(quality_scores)) * 100,
                'trusted_source_rate': len([s for s in sources if any(ts.lower() in s.lower() for ts in self.trusted_sources)]) / max(1, len(sources)) * 100
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating validation stats: {e}")
            return {'error': str(e)}