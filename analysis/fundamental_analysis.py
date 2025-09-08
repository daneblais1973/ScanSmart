import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import json

from shared.models import Catalyst, CatalystType
from core.config import AppConfig

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Advanced fundamental analysis for long-term investment signals"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Fundamental scoring weights
        self.fundamental_weights = {
            'profitability': 0.25,
            'growth': 0.20,
            'valuation': 0.20,
            'financial_health': 0.15,
            'efficiency': 0.10,
            'dividend': 0.10
        }
        
        # Industry benchmark multipliers
        self.industry_multipliers = {
            'Technology': {'growth': 1.3, 'valuation': 0.8},
            'Healthcare': {'growth': 1.2, 'valuation': 0.9},
            'Financial Services': {'dividend': 1.4, 'valuation': 1.1},
            'Utilities': {'dividend': 1.5, 'growth': 0.7},
            'Energy': {'volatility': 1.3, 'valuation': 1.2},
            'Consumer Cyclical': {'growth': 1.1, 'efficiency': 1.2},
            'Consumer Defensive': {'dividend': 1.3, 'stability': 1.4}
        }
        
        logger.info("Fundamental Analyzer initialized")
    
    async def analyze_stock_fundamentals(self, ticker: str, catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Comprehensive fundamental analysis for a stock"""
        try:
            # Get fundamental data
            financial_data = await self._get_financial_data(ticker)
            if not financial_data:
                logger.warning(f"No fundamental data available for {ticker}")
                return {}
            
            # Calculate financial ratios
            ratios = await self._calculate_financial_ratios(financial_data)
            
            # Analyze growth metrics
            growth_metrics = await self._analyze_growth_metrics(financial_data)
            
            # Evaluate valuation
            valuation_metrics = await self._analyze_valuation_metrics(financial_data, ticker)
            
            # Assess financial health
            health_metrics = await self._assess_financial_health(financial_data)
            
            # Calculate efficiency metrics
            efficiency_metrics = await self._calculate_efficiency_metrics(financial_data)
            
            # Analyze dividend metrics (if applicable)
            dividend_metrics = await self._analyze_dividend_metrics(financial_data)
            
            # Calculate overall fundamental score
            fundamental_score = await self._calculate_fundamental_score(
                ratios, growth_metrics, valuation_metrics, 
                health_metrics, efficiency_metrics, dividend_metrics,
                financial_data.get('sector', 'Unknown')
            )
            
            # Generate investment recommendations
            investment_signals = await self._generate_investment_signals(
                fundamental_score, ratios, growth_metrics, valuation_metrics, catalyst
            )
            
            return {
                'ticker': ticker,
                'sector': financial_data.get('sector', 'Unknown'),
                'market_cap': financial_data.get('marketCap', 0),
                'ratios': ratios,
                'growth_metrics': growth_metrics,
                'valuation_metrics': valuation_metrics,
                'health_metrics': health_metrics,
                'efficiency_metrics': efficiency_metrics,
                'dividend_metrics': dividend_metrics,
                'fundamental_score': fundamental_score,
                'investment_signals': investment_signals,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(financial_data)
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {ticker}: {e}")
            return {}
    
    async def _get_financial_data(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive financial data for a stock"""
        cache_key = f"fundamental_{ticker}"
        
        with self._cache_lock:
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                # Use cached data if less than 1 hour old for fundamentals
                if datetime.now() - timestamp < timedelta(hours=1):
                    return cached_data
        
        try:
            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(
                self._executor, 
                lambda: yf.Ticker(ticker)
            )
            
            # Get various financial data
            info = await loop.run_in_executor(self._executor, lambda: stock.info)
            financials = await loop.run_in_executor(self._executor, lambda: stock.financials)
            balance_sheet = await loop.run_in_executor(self._executor, lambda: stock.balance_sheet)
            cashflow = await loop.run_in_executor(self._executor, lambda: stock.cashflow)
            
            financial_data = {
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'enterpriseValue': info.get('enterpriseValue', 0)
            }
            
            if financial_data:
                with self._cache_lock:
                    self._cache[cache_key] = (financial_data, datetime.now())
                return financial_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {ticker}: {e}")
            return None
    
    async def _calculate_financial_ratios(self, financial_data: Dict) -> Dict[str, Any]:
        """Calculate key financial ratios"""
        try:
            ratios = {}
            info = financial_data.get('info', {})
            
            # Profitability ratios
            ratios['pe_ratio'] = info.get('trailingPE', 0) or info.get('forwardPE', 0)
            ratios['peg_ratio'] = info.get('pegRatio', 0)
            ratios['profit_margin'] = info.get('profitMargins', 0)
            ratios['operating_margin'] = info.get('operatingMargins', 0)
            ratios['gross_margin'] = info.get('grossMargins', 0)
            ratios['roe'] = info.get('returnOnEquity', 0)
            ratios['roa'] = info.get('returnOnAssets', 0)
            
            # Liquidity ratios
            ratios['current_ratio'] = info.get('currentRatio', 0)
            ratios['quick_ratio'] = info.get('quickRatio', 0)
            
            # Leverage ratios
            ratios['debt_to_equity'] = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            ratios['total_debt_to_capital'] = info.get('totalDebtToCapital', 0)
            
            # Efficiency ratios
            ratios['asset_turnover'] = info.get('assetTurnover', 0)
            ratios['inventory_turnover'] = info.get('inventoryTurnover', 0)
            
            # Market ratios
            ratios['price_to_book'] = info.get('priceToBook', 0)
            ratios['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
            ratios['ev_to_revenue'] = info.get('enterpriseToRevenue', 0)
            ratios['ev_to_ebitda'] = info.get('enterpriseToEbitda', 0)
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return {}
    
    async def _analyze_growth_metrics(self, financial_data: Dict) -> Dict[str, Any]:
        """Analyze growth metrics"""
        try:
            growth_metrics = {}
            info = financial_data.get('info', {})
            
            # Revenue growth
            growth_metrics['revenue_growth'] = info.get('revenueGrowth', 0)
            growth_metrics['revenue_growth_yoy'] = info.get('revenueQuarterlyGrowth', 0)
            
            # Earnings growth
            growth_metrics['earnings_growth'] = info.get('earningsGrowth', 0)
            growth_metrics['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', 0)
            
            # Other growth metrics
            growth_metrics['book_value_growth'] = info.get('bookValueGrowth', 0)
            growth_metrics['free_cashflow_growth'] = info.get('freeCashflowGrowth', 0)
            
            # Calculate growth consistency score
            growth_values = [
                growth_metrics['revenue_growth'],
                growth_metrics['earnings_growth']
            ]
            valid_growth = [g for g in growth_values if g and g != 0]
            
            if valid_growth:
                growth_metrics['growth_consistency'] = 1.0 - (np.std(valid_growth) / (np.mean(valid_growth) + 0.01))
                growth_metrics['average_growth'] = np.mean(valid_growth)
            else:
                growth_metrics['growth_consistency'] = 0.0
                growth_metrics['average_growth'] = 0.0
            
            return growth_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing growth metrics: {e}")
            return {}
    
    async def _analyze_valuation_metrics(self, financial_data: Dict, ticker: str) -> Dict[str, Any]:
        """Analyze valuation metrics with industry comparison"""
        try:
            valuation_metrics = {}
            info = financial_data.get('info', {})
            sector = financial_data.get('sector', 'Unknown')
            
            # Get current valuation ratios
            pe = info.get('trailingPE', 0) or info.get('forwardPE', 0)
            pb = info.get('priceToBook', 0)
            ps = info.get('priceToSalesTrailing12Months', 0)
            ev_revenue = info.get('enterpriseToRevenue', 0)
            
            valuation_metrics['pe_ratio'] = pe
            valuation_metrics['pb_ratio'] = pb
            valuation_metrics['ps_ratio'] = ps
            valuation_metrics['ev_revenue'] = ev_revenue
            
            # Industry comparison (simplified - in practice, you'd use external data)
            industry_benchmarks = {
                'Technology': {'pe': 25, 'pb': 4, 'ps': 6},
                'Healthcare': {'pe': 20, 'pb': 3, 'ps': 4},
                'Financial Services': {'pe': 12, 'pb': 1.2, 'ps': 3},
                'Utilities': {'pe': 16, 'pb': 1.5, 'ps': 2},
                'Energy': {'pe': 15, 'pb': 1.8, 'ps': 1.5},
                'Consumer Cyclical': {'pe': 18, 'pb': 2.5, 'ps': 1.8},
                'Consumer Defensive': {'pe': 22, 'pb': 3, 'ps': 1.5}
            }
            
            benchmark = industry_benchmarks.get(sector, {'pe': 20, 'pb': 2.5, 'ps': 3})
            
            # Calculate valuation scores (lower is better for valuation)
            valuation_scores = []
            
            if pe > 0:
                pe_score = max(0, 1 - (pe - benchmark['pe']) / benchmark['pe'])
                valuation_scores.append(pe_score)
                valuation_metrics['pe_vs_industry'] = pe / benchmark['pe'] if benchmark['pe'] > 0 else 1
            
            if pb > 0:
                pb_score = max(0, 1 - (pb - benchmark['pb']) / benchmark['pb'])
                valuation_scores.append(pb_score)
                valuation_metrics['pb_vs_industry'] = pb / benchmark['pb'] if benchmark['pb'] > 0 else 1
            
            if ps > 0:
                ps_score = max(0, 1 - (ps - benchmark['ps']) / benchmark['ps'])
                valuation_scores.append(ps_score)
                valuation_metrics['ps_vs_industry'] = ps / benchmark['ps'] if benchmark['ps'] > 0 else 1
            
            valuation_metrics['valuation_score'] = np.mean(valuation_scores) if valuation_scores else 0.5
            
            # Determine if stock is undervalued, fairly valued, or overvalued
            if valuation_metrics['valuation_score'] > 0.7:
                valuation_metrics['valuation_category'] = 'undervalued'
            elif valuation_metrics['valuation_score'] > 0.3:
                valuation_metrics['valuation_category'] = 'fairly_valued'
            else:
                valuation_metrics['valuation_category'] = 'overvalued'
            
            return valuation_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing valuation metrics: {e}")
            return {}
    
    async def _assess_financial_health(self, financial_data: Dict) -> Dict[str, Any]:
        """Assess financial health and stability"""
        try:
            health_metrics = {}
            info = financial_data.get('info', {})
            
            # Liquidity
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)
            
            health_metrics['liquidity_score'] = min(1.0, (current_ratio / 2.0)) if current_ratio else 0
            health_metrics['quick_liquidity_score'] = min(1.0, (quick_ratio / 1.5)) if quick_ratio else 0
            
            # Debt management
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            total_debt_to_capital = info.get('totalDebtToCapital', 0)
            
            # Lower debt ratios are better
            health_metrics['debt_management_score'] = max(0, 1 - (debt_to_equity / 2.0)) if debt_to_equity else 0.8
            
            # Cash position
            total_cash = info.get('totalCash', 0)
            total_debt = info.get('totalDebt', 0)
            market_cap = info.get('marketCap', 1)
            
            if market_cap > 0:
                health_metrics['cash_to_market_cap'] = total_cash / market_cap
                health_metrics['net_cash_position'] = (total_cash - total_debt) / market_cap
            else:
                health_metrics['cash_to_market_cap'] = 0
                health_metrics['net_cash_position'] = 0
            
            # Interest coverage
            ebitda = info.get('ebitda', 0)
            interest_expense = info.get('interestExpense', 0)
            
            if interest_expense and interest_expense < 0:  # Interest expense is usually negative
                health_metrics['interest_coverage'] = abs(ebitda / interest_expense) if ebitda else 0
            else:
                health_metrics['interest_coverage'] = float('inf') if ebitda > 0 else 0
            
            # Overall financial health score
            health_components = [
                health_metrics.get('liquidity_score', 0) * 0.3,
                health_metrics.get('debt_management_score', 0) * 0.4,
                min(1.0, health_metrics.get('interest_coverage', 0) / 5.0) * 0.3
            ]
            
            health_metrics['overall_health_score'] = sum(health_components)
            
            # Health category
            if health_metrics['overall_health_score'] > 0.8:
                health_metrics['health_category'] = 'excellent'
            elif health_metrics['overall_health_score'] > 0.6:
                health_metrics['health_category'] = 'good'
            elif health_metrics['overall_health_score'] > 0.4:
                health_metrics['health_category'] = 'fair'
            else:
                health_metrics['health_category'] = 'poor'
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error assessing financial health: {e}")
            return {}
    
    async def _calculate_efficiency_metrics(self, financial_data: Dict) -> Dict[str, Any]:
        """Calculate operational efficiency metrics"""
        try:
            efficiency_metrics = {}
            info = financial_data.get('info', {})
            
            # Margin efficiency
            gross_margin = info.get('grossMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            profit_margin = info.get('profitMargins', 0)
            
            efficiency_metrics['gross_margin'] = gross_margin
            efficiency_metrics['operating_margin'] = operating_margin
            efficiency_metrics['profit_margin'] = profit_margin
            
            # Margin progression (operating should be less than gross, profit less than operating)
            margin_efficiency = 0
            if gross_margin > 0:
                margin_efficiency += 0.4
                if operating_margin > 0 and operating_margin < gross_margin:
                    margin_efficiency += 0.3
                if profit_margin > 0 and profit_margin < operating_margin:
                    margin_efficiency += 0.3
            
            efficiency_metrics['margin_efficiency_score'] = margin_efficiency
            
            # Asset utilization
            roa = info.get('returnOnAssets', 0)
            roe = info.get('returnOnEquity', 0)
            asset_turnover = info.get('assetTurnover', 0)
            
            efficiency_metrics['roa'] = roa
            efficiency_metrics['roe'] = roe
            efficiency_metrics['asset_turnover'] = asset_turnover
            
            # Overall efficiency score
            efficiency_components = [
                min(1.0, gross_margin * 2) * 0.3 if gross_margin else 0,
                min(1.0, operating_margin * 4) * 0.3 if operating_margin else 0,
                min(1.0, roa * 5) * 0.2 if roa else 0,
                min(1.0, roe * 2) * 0.2 if roe else 0
            ]
            
            efficiency_metrics['overall_efficiency_score'] = sum(efficiency_components)
            
            return efficiency_metrics
            
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
            return {}
    
    async def _analyze_dividend_metrics(self, financial_data: Dict) -> Dict[str, Any]:
        """Analyze dividend metrics for income-focused investing"""
        try:
            dividend_metrics = {}
            info = financial_data.get('info', {})
            
            # Basic dividend info
            dividend_yield = info.get('dividendYield', 0)
            dividend_rate = info.get('dividendRate', 0)
            payout_ratio = info.get('payoutRatio', 0)
            
            dividend_metrics['dividend_yield'] = dividend_yield
            dividend_metrics['dividend_rate'] = dividend_rate
            dividend_metrics['payout_ratio'] = payout_ratio
            
            # Dividend sustainability
            if payout_ratio and 0 < payout_ratio < 0.8:  # Sustainable payout ratio
                dividend_metrics['dividend_sustainability'] = 'sustainable'
                dividend_metrics['sustainability_score'] = 1.0 - payout_ratio
            elif payout_ratio and payout_ratio >= 0.8:
                dividend_metrics['dividend_sustainability'] = 'at_risk'
                dividend_metrics['sustainability_score'] = max(0, 1.2 - payout_ratio)
            elif payout_ratio == 0:
                dividend_metrics['dividend_sustainability'] = 'no_dividend'
                dividend_metrics['sustainability_score'] = 0
            else:
                dividend_metrics['dividend_sustainability'] = 'unknown'
                dividend_metrics['sustainability_score'] = 0.5
            
            # Dividend attractiveness score
            if dividend_yield:
                # Good yield is context-dependent, but generally 2-6% is attractive
                yield_score = min(1.0, dividend_yield / 0.04) * 0.6  # Normalize to 4% yield
                sustainability_score = dividend_metrics.get('sustainability_score', 0) * 0.4
                dividend_metrics['dividend_attractiveness'] = yield_score + sustainability_score
            else:
                dividend_metrics['dividend_attractiveness'] = 0
            
            return dividend_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing dividend metrics: {e}")
            return {}
    
    async def _calculate_fundamental_score(self, ratios: Dict, growth_metrics: Dict, 
                                         valuation_metrics: Dict, health_metrics: Dict,
                                         efficiency_metrics: Dict, dividend_metrics: Dict,
                                         sector: str) -> Dict[str, Any]:
        """Calculate overall fundamental score with sector adjustments"""
        try:
            # Component scores
            profitability_score = min(1.0, (
                (ratios.get('roe', 0) * 2) * 0.4 +
                (ratios.get('roa', 0) * 4) * 0.3 +
                (ratios.get('profit_margin', 0) * 3) * 0.3
            )) if any([ratios.get('roe'), ratios.get('roa'), ratios.get('profit_margin')]) else 0
            
            growth_score = min(1.0, (
                max(0, min(1, growth_metrics.get('revenue_growth', 0) * 2)) * 0.4 +
                max(0, min(1, growth_metrics.get('earnings_growth', 0) * 2)) * 0.4 +
                growth_metrics.get('growth_consistency', 0) * 0.2
            ))
            
            valuation_score = valuation_metrics.get('valuation_score', 0.5)
            health_score = health_metrics.get('overall_health_score', 0.5)
            efficiency_score = efficiency_metrics.get('overall_efficiency_score', 0.5)
            dividend_score = dividend_metrics.get('dividend_attractiveness', 0)
            
            # Apply sector multipliers
            if sector in self.industry_multipliers:
                multipliers = self.industry_multipliers[sector]
                growth_score *= multipliers.get('growth', 1.0)
                valuation_score *= multipliers.get('valuation', 1.0)
                dividend_score *= multipliers.get('dividend', 1.0)
            
            # Calculate weighted overall score
            overall_score = (
                profitability_score * self.fundamental_weights['profitability'] +
                growth_score * self.fundamental_weights['growth'] +
                valuation_score * self.fundamental_weights['valuation'] +
                health_score * self.fundamental_weights['financial_health'] +
                efficiency_score * self.fundamental_weights['efficiency'] +
                dividend_score * self.fundamental_weights['dividend']
            )
            
            return {
                'overall_score': min(1.0, overall_score),
                'component_scores': {
                    'profitability': profitability_score,
                    'growth': growth_score,
                    'valuation': valuation_score,
                    'financial_health': health_score,
                    'efficiency': efficiency_score,
                    'dividend': dividend_score
                },
                'sector_adjusted': sector in self.industry_multipliers
            }
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return {'overall_score': 0.5, 'component_scores': {}, 'sector_adjusted': False}
    
    async def _generate_investment_signals(self, fundamental_score: Dict, ratios: Dict,
                                         growth_metrics: Dict, valuation_metrics: Dict,
                                         catalyst: Optional[Catalyst] = None) -> Dict[str, Any]:
        """Generate investment signals based on fundamental analysis"""
        try:
            signals = {
                'long_term_rating': 'HOLD',
                'confidence': 0.0,
                'target_horizon': 'long_term',
                'key_strengths': [],
                'key_concerns': [],
                'catalyst_impact': 'none'
            }
            
            overall_score = fundamental_score.get('overall_score', 0.5)
            component_scores = fundamental_score.get('component_scores', {})
            
            # Determine rating based on overall score
            if overall_score >= 0.8:
                signals['long_term_rating'] = 'STRONG BUY'
                signals['confidence'] = 0.9
            elif overall_score >= 0.65:
                signals['long_term_rating'] = 'BUY'
                signals['confidence'] = 0.8
            elif overall_score >= 0.35:
                signals['long_term_rating'] = 'HOLD'
                signals['confidence'] = 0.6
            elif overall_score >= 0.2:
                signals['long_term_rating'] = 'SELL'
                signals['confidence'] = 0.7
            else:
                signals['long_term_rating'] = 'STRONG SELL'
                signals['confidence'] = 0.8
            
            # Identify key strengths
            if component_scores.get('profitability', 0) > 0.7:
                signals['key_strengths'].append('Strong profitability metrics')
            if component_scores.get('growth', 0) > 0.7:
                signals['key_strengths'].append('Excellent growth trajectory')
            if component_scores.get('valuation', 0) > 0.7:
                signals['key_strengths'].append('Attractive valuation')
            if component_scores.get('financial_health', 0) > 0.8:
                signals['key_strengths'].append('Excellent financial health')
            if component_scores.get('dividend', 0) > 0.6:
                signals['key_strengths'].append('Strong dividend profile')
            
            # Identify key concerns
            if component_scores.get('profitability', 0) < 0.3:
                signals['key_concerns'].append('Weak profitability')
            if component_scores.get('growth', 0) < 0.2:
                signals['key_concerns'].append('Poor growth prospects')
            if component_scores.get('valuation', 0) < 0.3:
                signals['key_concerns'].append('Potentially overvalued')
            if component_scores.get('financial_health', 0) < 0.4:
                signals['key_concerns'].append('Financial health concerns')
            
            # Catalyst impact
            if catalyst:
                if catalyst.category in [CatalystType.EARNINGS, CatalystType.ANALYST] and catalyst.impact >= 80:
                    signals['catalyst_impact'] = 'positive' if catalyst.sentiment_score > 0 else 'negative'
                    # Adjust confidence based on catalyst
                    catalyst_adjustment = (catalyst.impact / 100.0) * 0.1
                    if catalyst.sentiment_score > 0 and signals['long_term_rating'] in ['BUY', 'STRONG BUY']:
                        signals['confidence'] = min(1.0, signals['confidence'] + catalyst_adjustment)
                    elif catalyst.sentiment_score < 0 and signals['long_term_rating'] in ['SELL', 'STRONG SELL']:
                        signals['confidence'] = min(1.0, signals['confidence'] + catalyst_adjustment)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating investment signals: {e}")
            return {
                'long_term_rating': 'HOLD',
                'confidence': 0.5,
                'target_horizon': 'long_term',
                'key_strengths': [],
                'key_concerns': [],
                'catalyst_impact': 'none'
            }
    
    def _assess_data_quality(self, financial_data: Dict) -> str:
        """Assess the quality and completeness of financial data"""
        try:
            info = financial_data.get('info', {})
            
            # Check for key data points
            key_metrics = [
                'trailingPE', 'forwardPE', 'priceToBook', 'profitMargins',
                'returnOnEquity', 'revenueGrowth', 'earningsGrowth', 'currentRatio'
            ]
            
            available_metrics = sum(1 for metric in key_metrics if info.get(metric) is not None)
            completeness_ratio = available_metrics / len(key_metrics)
            
            if completeness_ratio >= 0.8:
                return 'high'
            elif completeness_ratio >= 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 'unknown'
    
    async def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
        logger.info("Fundamental Analyzer cleaned up")
