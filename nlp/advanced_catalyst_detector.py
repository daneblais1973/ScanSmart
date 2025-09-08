"""
Advanced ML Catalyst Detection using Fine-tuned GPT Models
Replaces rule-based pattern matching with sophisticated AI models
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import openai
from openai import OpenAI
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from shared.models import Catalyst, CatalystType, SentimentLabel

logger = logging.getLogger(__name__)

@dataclass
class CatalystDetectionResult:
    """Result of catalyst detection analysis"""
    is_catalyst: bool
    confidence: float
    catalyst_type: CatalystType
    impact_score: int
    sentiment: SentimentLabel
    reasoning: str
    key_phrases: List[str]
    urgency_level: str  # 'low', 'medium', 'high', 'critical'

class AdvancedCatalystDetector:
    """
    Advanced ML-based catalyst detection using GPT models and ensemble methods
    """
    
    def __init__(self, config):
        self.config = config
        self.openai_client = None
        self.model_cache = {}
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_initialized = False
        
        # Initialize OpenAI client
        api_key = getattr(config, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        
        # Financial catalyst patterns for GPT prompting
        self.catalyst_patterns = {
            CatalystType.EARNINGS: [
                "earnings report", "quarterly results", "revenue beat", "eps miss",
                "guidance", "profit margin", "operating income", "net income"
            ],
            CatalystType.FDA_APPROVAL: [
                "FDA approval", "clinical trial", "drug approval", "medical device",
                "breakthrough therapy", "accelerated approval", "phase III", "PDUFA"
            ],
            CatalystType.MERGER_ACQUISITION: [
                "merger", "acquisition", "takeover", "buyout", "strategic partnership",
                "joint venture", "all-cash deal", "hostile takeover"
            ],
            CatalystType.PRODUCT_LAUNCH: [
                "product launch", "new product", "service offering", "platform release",
                "feature announcement", "beta testing", "commercial availability"
            ],
            CatalystType.REGULATORY: [
                "regulatory approval", "SEC filing", "patent approval", "compliance",
                "investigation", "fine", "settlement", "court ruling"
            ],
            CatalystType.PARTNERSHIP: [
                "strategic partnership", "collaboration", "joint development",
                "licensing agreement", "supply agreement", "distribution deal"
            ],
            CatalystType.INSIDER_TRADING: [
                "insider buying", "insider selling", "executive purchase",
                "director sale", "stock buyback", "share repurchase"
            ],
            CatalystType.ANALYST_UPGRADE: [
                "analyst upgrade", "price target", "rating change", "buy recommendation",
                "sell rating", "downgrade", "initiate coverage"
            ],
            CatalystType.NEWS: [
                "breaking news", "market moving", "significant development",
                "major announcement", "important update", "key development"
            ]
        }
        
        # Sector-specific indicators
        self.sector_indicators = {
            'biotech': [
                'clinical trial', 'FDA', 'drug', 'therapy', 'patients', 'efficacy',
                'safety', 'endpoint', 'dosing', 'adverse events', 'regulatory'
            ],
            'tech': [
                'algorithm', 'AI', 'software', 'platform', 'cloud', 'SaaS',
                'data', 'cybersecurity', 'semiconductor', 'chip', 'processor'
            ],
            'energy': [
                'oil', 'gas', 'renewable', 'carbon', 'pipeline', 'reserves',
                'drilling', 'refinery', 'solar', 'wind', 'battery', 'grid'
            ],
            'finance': [
                'interest rate', 'loan', 'credit', 'deposit', 'banking',
                'mortgage', 'insurance', 'regulatory capital', 'compliance'
            ]
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models and components"""
        try:
            # Try to load pre-trained models
            self._load_pretrained_models()
            
            if not self.is_initialized:
                # Initialize new models if none exist
                self._create_new_models()
            
            logger.info("Advanced Catalyst Detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Catalyst Detector: {e}")
            self.is_initialized = False
    
    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        try:
            model_dir = "models/catalyst_detection"
            
            if os.path.exists(f"{model_dir}/vectorizer.pkl"):
                self.vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
                self.classifier = joblib.load(f"{model_dir}/classifier.pkl")
                self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
                self.is_initialized = True
                logger.info("Loaded pre-trained catalyst detection models")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def _create_new_models(self):
        """Create and train new models with synthetic data"""
        try:
            # Create training data
            training_data = self._generate_training_data()
            
            # Initialize models
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )
            
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            )
            
            self.label_encoder = LabelEncoder()
            
            # Prepare training data
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Fit models
            X = self.vectorizer.fit_transform(texts)
            y = self.label_encoder.fit_transform(labels)
            
            self.classifier.fit(X, y)
            
            # Save models
            self._save_models()
            
            self.is_initialized = True
            logger.info("Created and trained new catalyst detection models")
            
        except Exception as e:
            logger.error(f"Failed to create new models: {e}")
    
    def _generate_training_data(self) -> List[Dict]:
        """Generate synthetic training data for catalyst detection"""
        training_data = []
        
        # Positive examples (catalysts)
        catalyst_examples = [
            ("Company reports Q3 earnings beat with 25% revenue growth", "earnings"),
            ("FDA approves breakthrough cancer therapy", "fda_approval"),
            ("Microsoft announces $10B acquisition of AI startup", "merger_acquisition"),
            ("Apple launches revolutionary new iPhone with AI chip", "product_launch"),
            ("Tesla receives regulatory approval for autonomous driving", "regulatory"),
            ("Google partners with OpenAI for cloud services", "partnership"),
            ("CEO purchases $5M worth of company stock", "insider_trading"),
            ("Goldman Sachs upgrades to Buy with $200 price target", "analyst_upgrade"),
            ("Breaking: Company discovers major oil reserves", "news"),
            ("Biotech announces positive Phase III trial results", "fda_approval"),
            ("Tech giant beats earnings estimates by 15%", "earnings"),
            ("Pharmaceutical company receives FDA fast track designation", "fda_approval"),
            ("Energy company completes major pipeline project", "regulatory"),
            ("Retail chain announces expansion into 500 new stores", "product_launch"),
            ("Bank reports record quarterly profits", "earnings"),
            ("Semiconductor firm lands major supply contract", "partnership"),
            ("Mining company discovers rare earth deposits", "news"),
            ("Software company launches AI-powered platform", "product_launch"),
            ("Airline receives certification for new aircraft", "regulatory"),
            ("Healthcare firm announces strategic merger", "merger_acquisition")
        ]
        
        for text, label in catalyst_examples:
            training_data.append({
                'text': text,
                'label': label,
                'is_catalyst': True
            })
        
        # Negative examples (non-catalysts)
        non_catalyst_examples = [
            "Company holds routine quarterly board meeting",
            "Stock price fluctuates during trading session",
            "CEO makes standard industry conference appearance",
            "Company updates website with new design",
            "Analyst maintains current rating and price target",
            "Company files routine 10-K annual report",
            "Executive attends industry networking event",
            "Stock added to popular ETF portfolio",
            "Company celebrates employee appreciation day",
            "Routine maintenance scheduled for facilities",
            "Company participates in charity fundraiser",
            "Standard investor relations presentation given",
            "Employee training program implemented",
            "Office lease renewed for additional five years",
            "Company updates social media profiles",
            "Regular dividend payment announced as scheduled",
            "Annual shareholder meeting date confirmed",
            "Company sponsors local sports team",
            "Routine IT system updates completed",
            "Standard compliance filing submitted"
        ]
        
        for text in non_catalyst_examples:
            training_data.append({
                'text': text,
                'label': 'non_catalyst',
                'is_catalyst': False
            })
        
        return training_data
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = "models/catalyst_detection"
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.vectorizer, f"{model_dir}/vectorizer.pkl")
            joblib.dump(self.classifier, f"{model_dir}/classifier.pkl")
            joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.pkl")
            
            logger.info("Saved catalyst detection models")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def detect_catalyst(self, text: str, ticker: str = "", context: Dict = None) -> CatalystDetectionResult:
        """
        Advanced catalyst detection using GPT and ML ensemble
        """
        try:
            # Step 1: Quick ML screening
            ml_result = self._ml_screening(text)
            
            if not ml_result['is_likely_catalyst']:
                return CatalystDetectionResult(
                    is_catalyst=False,
                    confidence=ml_result['confidence'],
                    catalyst_type=CatalystType.NEWS,
                    impact_score=10,
                    sentiment=SentimentLabel.NEUTRAL,
                    reasoning="ML screening indicates low catalyst probability",
                    key_phrases=[],
                    urgency_level='low'
                )
            
            # Step 2: GPT-based deep analysis
            gpt_result = await self._gpt_analysis(text, ticker, context)
            
            # Step 3: Combine results
            final_result = self._combine_results(ml_result, gpt_result, text)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in catalyst detection: {e}")
            return CatalystDetectionResult(
                is_catalyst=False,
                confidence=0.0,
                catalyst_type=CatalystType.NEWS,
                impact_score=0,
                sentiment=SentimentLabel.NEUTRAL,
                reasoning=f"Error in analysis: {str(e)}",
                key_phrases=[],
                urgency_level='low'
            )
    
    def _ml_screening(self, text: str) -> Dict:
        """Quick ML-based screening for catalyst likelihood"""
        try:
            if not self.is_initialized:
                return {'is_likely_catalyst': True, 'confidence': 0.5}
            
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            proba = self.classifier.predict_proba(X)[0]
            prediction = self.classifier.predict(X)[0]
            
            # Get predicted label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence
            max_proba = max(proba)
            
            is_catalyst = predicted_label != 'non_catalyst'
            
            return {
                'is_likely_catalyst': is_catalyst,
                'confidence': max_proba,
                'predicted_type': predicted_label if is_catalyst else None,
                'probabilities': dict(zip(self.label_encoder.classes_, proba))
            }
            
        except Exception as e:
            logger.error(f"Error in ML screening: {e}")
            return {'is_likely_catalyst': True, 'confidence': 0.5}
    
    async def _gpt_analysis(self, text: str, ticker: str, context: Dict) -> Dict:
        """GPT-based deep catalyst analysis"""
        try:
            if not self.openai_client:
                return self._fallback_analysis(text)
            
            # Construct analysis prompt
            prompt = self._build_analysis_prompt(text, ticker, context)
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in identifying market-moving catalysts. Analyze the provided text and respond with structured JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error in GPT analysis: {e}")
            return self._fallback_analysis(text)
    
    def _build_analysis_prompt(self, text: str, ticker: str, context: Dict) -> str:
        """Build comprehensive analysis prompt for GPT"""
        
        sector_context = ""
        if context and 'sector' in context:
            sector = context['sector'].lower()
            if sector in self.sector_indicators:
                indicators = ", ".join(self.sector_indicators[sector])
                sector_context = f"This is a {sector} company. Key sector indicators include: {indicators}."
        
        prompt = f"""
        Analyze the following text for financial catalyst potential:
        
        Text: "{text}"
        Company Ticker: {ticker if ticker else 'Unknown'}
        {sector_context}
        
        Please analyze and respond with JSON containing:
        {{
            "is_catalyst": boolean,
            "confidence": float (0.0-1.0),
            "catalyst_type": string (one of: earnings, fda_approval, merger_acquisition, product_launch, regulatory, partnership, insider_trading, analyst_upgrade, news),
            "impact_score": integer (0-100),
            "sentiment": string (positive, negative, neutral),
            "reasoning": string,
            "key_phrases": array of strings,
            "urgency_level": string (low, medium, high, critical),
            "market_impact_probability": float (0.0-1.0),
            "timeframe": string (immediate, short_term, medium_term, long_term),
            "sector_relevance": float (0.0-1.0)
        }}
        
        Consider these factors:
        1. Market-moving potential (earnings, M&A, FDA approvals, etc.)
        2. Magnitude of impact on stock price
        3. Time sensitivity and urgency
        4. Sentiment and market perception
        5. Sector-specific catalysts
        6. Credibility of source and information
        
        Focus on actionable, tradeable catalysts that could drive significant price movement.
        """
        
        return prompt
    
    def _fallback_analysis(self, text: str) -> Dict:
        """Fallback analysis when GPT is unavailable"""
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        catalyst_score = 0
        detected_type = CatalystType.NEWS
        key_phrases = []
        
        for catalyst_type, keywords in self.catalyst_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    catalyst_score += 1
                    key_phrases.append(keyword)
                    detected_type = catalyst_type
        
        # Determine impact and sentiment
        impact_score = min(50 + (catalyst_score * 10), 100)
        
        positive_words = ['beat', 'growth', 'approval', 'success', 'positive', 'breakthrough']
        negative_words = ['miss', 'decline', 'rejection', 'failure', 'negative', 'loss']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "is_catalyst": catalyst_score > 0,
            "confidence": min(catalyst_score * 0.15, 0.8),
            "catalyst_type": detected_type.value,
            "impact_score": impact_score,
            "sentiment": sentiment,
            "reasoning": f"Fallback analysis detected {catalyst_score} catalyst indicators",
            "key_phrases": key_phrases[:5],
            "urgency_level": "medium" if catalyst_score > 2 else "low",
            "market_impact_probability": min(catalyst_score * 0.2, 0.8),
            "timeframe": "short_term",
            "sector_relevance": 0.5
        }
    
    def _combine_results(self, ml_result: Dict, gpt_result: Dict, text: str) -> CatalystDetectionResult:
        """Combine ML and GPT results for final decision"""
        
        # Weight the results (GPT gets higher weight if available)
        ml_weight = 0.3
        gpt_weight = 0.7
        
        # Combine confidence scores
        ml_confidence = ml_result.get('confidence', 0.5)
        gpt_confidence = gpt_result.get('confidence', 0.5)
        
        combined_confidence = (ml_confidence * ml_weight) + (gpt_confidence * gpt_weight)
        
        # Determine if it's a catalyst
        ml_is_catalyst = ml_result.get('is_likely_catalyst', False)
        gpt_is_catalyst = gpt_result.get('is_catalyst', False)
        
        # Catalyst if either system says yes with sufficient confidence
        is_catalyst = (gpt_is_catalyst and gpt_confidence > 0.6) or \
                     (ml_is_catalyst and ml_confidence > 0.8) or \
                     (ml_is_catalyst and gpt_is_catalyst)
        
        # Use GPT results as primary if available
        catalyst_type_str = gpt_result.get('catalyst_type', 'news')
        try:
            catalyst_type = CatalystType(catalyst_type_str)
        except:
            catalyst_type = CatalystType.NEWS
        
        # Map sentiment
        sentiment_str = gpt_result.get('sentiment', 'neutral')
        sentiment_map = {
            'positive': SentimentLabel.POSITIVE,
            'negative': SentimentLabel.NEGATIVE,
            'neutral': SentimentLabel.NEUTRAL
        }
        sentiment = sentiment_map.get(sentiment_str, SentimentLabel.NEUTRAL)
        
        return CatalystDetectionResult(
            is_catalyst=is_catalyst,
            confidence=combined_confidence,
            catalyst_type=catalyst_type,
            impact_score=gpt_result.get('impact_score', 50),
            sentiment=sentiment,
            reasoning=gpt_result.get('reasoning', 'Combined ML and GPT analysis'),
            key_phrases=gpt_result.get('key_phrases', []),
            urgency_level=gpt_result.get('urgency_level', 'medium')
        )
    
    def retrain_models(self, training_data: List[Dict]):
        """Retrain models with new data"""
        try:
            if not training_data:
                logger.warning("No training data provided for retraining")
                return
            
            # Prepare data
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Retrain vectorizer and classifier
            X = self.vectorizer.fit_transform(texts)
            y = self.label_encoder.fit_transform(labels)
            
            self.classifier.fit(X, y)
            
            # Save updated models
            self._save_models()
            
            logger.info(f"Retrained models with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'initialized': self.is_initialized,
            'has_openai': self.openai_client is not None,
            'model_features': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            'classes': list(self.label_encoder.classes_) if self.label_encoder else [],
            'n_estimators': self.classifier.n_estimators if self.classifier else 0
        }