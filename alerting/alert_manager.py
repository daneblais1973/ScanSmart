import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import threading
import json

from .notification_channels import EmailChannel, SMSChannel, ConsoleChannel, WebhookChannel
from shared.models import Catalyst, AlertRule
from analysis.opportunity_scorer import TradingOpportunity
from core.config import AppConfig

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages real-time alerting for financial catalysts"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize notification channels
        self.channels = {}
        self._init_channels()
        
        # Rate limiting and deduplication
        self._alert_history = {}
        self._rate_limit_lock = threading.Lock()
        
        # Alert rules - load default professional screening rules
        self.alert_rules = self._load_default_alert_rules()
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'alerts_sent': 0,
            'alerts_filtered': 0,
            'alerts_failed': 0,
            'last_alert': None
        }
        
        logger.info(f"Alert Manager initialized with {len(self.channels)} notification channels")
    
    def _init_channels(self):
        """Initialize notification channels based on configuration"""
        try:
            # Console channel (always available)
            self.channels['console'] = ConsoleChannel()
            
            # Email channel
            if (self.config.api.smtp_server and 
                self.config.api.smtp_username and 
                self.config.api.smtp_password):
                self.channels['email'] = EmailChannel(
                    smtp_server=self.config.api.smtp_server,
                    smtp_port=self.config.api.smtp_port,
                    username=self.config.api.smtp_username,
                    password=self.config.api.smtp_password
                )
                logger.info("Email channel initialized")
            
            # SMS channel (Twilio)
            if (self.config.api.twilio_account_sid and 
                self.config.api.twilio_auth_token and 
                self.config.api.twilio_phone_number):
                self.channels['sms'] = SMSChannel(
                    account_sid=self.config.api.twilio_account_sid,
                    auth_token=self.config.api.twilio_auth_token,
                    from_phone=self.config.api.twilio_phone_number
                )
                logger.info("SMS channel initialized")
            
            # Webhook channel
            self.channels['webhook'] = WebhookChannel()
            
            logger.info(f"Initialized {len(self.channels)} notification channels")
            
        except Exception as e:
            logger.error(f"Error initializing notification channels: {e}")
    
    def _load_default_alert_rules(self) -> List[AlertRule]:
        """Load default professional-grade alert rules for stock screening"""
        try:
            # Create professional alert rules similar to institutional screeners
            default_rules = [
                # High-impact catalysts (always alert)
                AlertRule(
                    name="High Impact Catalysts",
                    min_impact_score=85,
                    max_impact_score=100,
                    categories=['earnings', 'ma', 'regulatory'],
                    min_confidence=0.7,
                    enabled=True
                ),
                
                # Medium-impact with high confidence
                AlertRule(
                    name="Medium Impact High Confidence",
                    min_impact_score=70,
                    max_impact_score=85,
                    categories=['earnings', 'ma', 'regulatory', 'analyst'],
                    min_confidence=0.8,
                    enabled=True
                ),
                
                # Insider trading alerts
                AlertRule(
                    name="Insider Trading Activity",
                    min_impact_score=60,
                    max_impact_score=100,
                    categories=['insider'],
                    min_confidence=0.6,
                    enabled=True
                ),
                
                # M&A and takeover rumors (lower threshold)
                AlertRule(
                    name="M&A Activity",
                    min_impact_score=75,
                    max_impact_score=100,
                    categories=['ma'],
                    min_confidence=0.5,  # Lower confidence OK for M&A rumors
                    enabled=True
                ),
                
                # FDA decisions and clinical trials
                AlertRule(
                    name="Regulatory Decisions",
                    min_impact_score=80,
                    max_impact_score=100,
                    categories=['regulatory'],
                    min_confidence=0.7,
                    enabled=True
                )
            ]
            
            logger.info(f"Loaded {len(default_rules)} default alert rules")
            return default_rules
            
        except Exception as e:
            logger.error(f"Error loading default alert rules: {e}")
            return []
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: min_impact={rule.min_impact_score}")
    
    def remove_alert_rule(self, rule: AlertRule):
        """Remove an alert rule"""
        if rule in self.alert_rules:
            self.alert_rules.remove(rule)
            logger.info("Alert rule removed")
    
    async def process_catalyst(self, catalyst: Catalyst) -> Dict[str, Any]:
        """Process a catalyst and send alerts if criteria are met"""
        try:
            self._stats['total_alerts'] += 1
            
            # Check if catalyst meets alert criteria
            if not await self._should_alert(catalyst):
                self._stats['alerts_filtered'] += 1
                logger.debug(f"Catalyst filtered: {catalyst.ticker} - {catalyst.impact}")
                return {
                    'alert_sent': False,
                    'reason': 'filtered',
                    'catalyst_id': catalyst.ticker
                }
            
            # Check rate limiting
            if await self._is_rate_limited(catalyst):
                self._stats['alerts_filtered'] += 1
                logger.debug(f"Catalyst rate limited: {catalyst.ticker}")
                return {
                    'alert_sent': False,
                    'reason': 'rate_limited',
                    'catalyst_id': catalyst.ticker
                }
            
            # Send alerts
            results = await self._send_alerts(catalyst)
            
            if any(result.get('success', False) for result in results.values()):
                self._stats['alerts_sent'] += 1
                self._stats['last_alert'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Alert sent for catalyst: {catalyst.ticker} - {catalyst.category}")
                
                # Update rate limiting history
                await self._update_alert_history(catalyst)
                
                return {
                    'alert_sent': True,
                    'catalyst_id': catalyst.ticker,
                    'results': results
                }
            else:
                self._stats['alerts_failed'] += 1
                logger.warning(f"All alert channels failed for: {catalyst.ticker}")
                return {
                    'alert_sent': False,
                    'reason': 'all_channels_failed',
                    'catalyst_id': catalyst.ticker,
                    'results': results
                }
            
        except Exception as e:
            logger.error(f"Error processing catalyst alert: {e}")
            self._stats['alerts_failed'] += 1
            return {
                'alert_sent': False,
                'reason': 'error',
                'error': str(e),
                'catalyst_id': getattr(catalyst, 'ticker', 'unknown')
            }
    
    def calculate_alert_priority_score(self, catalyst: Catalyst) -> float:
        """Calculate smart alert priority score based on multiple factors"""
        try:
            base_score = catalyst.impact
            
            # Catalyst type weights
            type_weights = {
                'earnings': 1.2,
                'ma': 1.3,  # M&A highest priority
                'regulatory': 1.1,
                'analyst': 0.9,
                'insider': 1.0,
                'general': 0.8
            }
            
            # Apply type weight
            type_weight = type_weights.get(catalyst.category.lower(), 1.0)
            weighted_score = base_score * type_weight
            
            # Confidence boost
            confidence_boost = catalyst.confidence * 0.2
            
            # Source credibility factor
            source_weights = {
                'NewsAPI': 1.0,
                'RSS': 0.9,
                'Twitter': 0.8,
                'Reddit': 0.7
            }
            source_weight = source_weights.get(catalyst.source, 0.8)
            
            # Recency factor (newer catalysts get higher priority)
            if catalyst.published_date:
                hours_old = (datetime.now(timezone.utc) - catalyst.published_date).total_seconds() / 3600
                recency_factor = max(0.5, 1.0 - (hours_old / 24))  # Decay over 24 hours
            else:
                recency_factor = 0.8
            
            # Market cap factor (larger companies get slight boost for stability)
            market_cap_factor = 1.0  # Would need market cap data
            
            # Final priority score
            priority_score = (weighted_score + confidence_boost) * source_weight * recency_factor * market_cap_factor
            
            return min(100.0, priority_score)
            
        except Exception as e:
            logger.error(f"Error calculating alert priority: {e}")
            return catalyst.impact

    async def _should_alert(self, catalyst: Catalyst) -> bool:
        """Enhanced alert criteria with smart prioritization"""
        try:
            # Calculate priority score
            priority_score = self.calculate_alert_priority_score(catalyst)
            
            # Dynamic threshold based on alert volume
            base_threshold = self.config.alerts.min_impact_score
            
            # Adjust threshold based on recent alert volume
            recent_alert_count = await self._get_recent_alert_count()
            if recent_alert_count > 10:  # High volume - raise threshold
                adjusted_threshold = base_threshold + 10
            elif recent_alert_count < 3:  # Low volume - lower threshold
                adjusted_threshold = max(50, base_threshold - 5)
            else:
                adjusted_threshold = base_threshold
            
            # Check if priority score meets threshold
            if priority_score < adjusted_threshold:
                return False
            
            # Enhanced confidence check with context
            min_confidence = self.config.alerts.min_confidence
            
            # Lower confidence requirements for high-impact catalysts
            if priority_score > 85:
                min_confidence *= 0.8
            elif priority_score > 95:
                min_confidence *= 0.7
                
            if catalyst.confidence < min_confidence:
                return False
            
            # Check custom alert rules
            if self.alert_rules:
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    # Check impact threshold
                    if catalyst.impact < rule.min_impact_score:
                        continue
                    
                    # Check ticker filter
                    if rule.tickers and catalyst.ticker not in rule.tickers:
                        continue
                    
                    # Check category filter
                    if rule.categories and catalyst.category not in rule.categories:
                        continue
                    
                    # If we reach here, rule matches
                    return True
                
                # If we have rules but none matched
                return False
            
            # Default: alert if meets basic criteria
            return True
            
        except Exception as e:
            logger.error(f"Error checking alert criteria: {e}")
            return False
    
    async def test_notification_system(self) -> Dict[str, Any]:
        """Test all notification channels to verify functionality"""
        try:
            test_results = {}
            
            # Create a test catalyst
            from shared.models import Catalyst, CatalystType, SentimentLabel, SourceType
            from datetime import datetime, timezone
            
            test_catalyst = Catalyst(
                ticker="TEST",
                catalyst="Test catalyst for notification system verification",
                category=CatalystType.GENERAL,
                sentiment_label=SentimentLabel.POSITIVE,
                sector="Technology",
                sentiment_score=0.8,
                impact=90,
                source=SourceType.RSS,
                confidence=0.9,
                published_date=datetime.now(timezone.utc),
                url="https://example.com/test",
                metadata={"test": True}
            )
            
            # Test each channel
            for channel_name, channel in self.channels.items():
                try:
                    result = await channel.send_alert(
                        message="This is a test alert to verify the notification system is working properly.",
                        subject="🔔 QuantumCatalyst Pro - Notification System Test",
                        catalyst=test_catalyst,
                        recipient=None
                    )
                    test_results[channel_name] = result
                    logger.info(f"Notification test - {channel_name}: {result.get('success', False)}")
                except Exception as e:
                    test_results[channel_name] = {
                        'success': False,
                        'error': str(e),
                        'channel': channel_name
                    }
                    logger.error(f"Notification test failed - {channel_name}: {e}")
            
            # Summary
            working_channels = [name for name, result in test_results.items() if result.get('success', False)]
            
            return {
                'test_completed': True,
                'total_channels': len(self.channels),
                'working_channels': len(working_channels),
                'working_channel_names': working_channels,
                'detailed_results': test_results,
                'overall_status': 'PASS' if working_channels else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Error testing notification system: {e}")
            return {
                'test_completed': False,
                'error': str(e),
                'overall_status': 'ERROR'
            }
    
    async def _get_recent_alert_count(self) -> int:
        """Get count of alerts sent in the last hour"""
        try:
            current_time = datetime.now(timezone.utc)
            one_hour_ago = current_time - timedelta(hours=1)
            
            # Count alerts from alert history
            recent_count = 0
            with self._rate_limit_lock:
                for key, alert_time in self._alert_history.items():
                    if alert_time > one_hour_ago:
                        recent_count += 1
            
            return recent_count
        except Exception as e:
            logger.error(f"Error getting recent alert count: {e}")
            return 0
    
    async def _is_rate_limited(self, catalyst: Catalyst) -> bool:
        """Check if catalyst is rate limited"""
        try:
            with self._rate_limit_lock:
                current_time = datetime.now(timezone.utc)
                rate_limit_window = timedelta(minutes=self.config.alerts.rate_limit_minutes)
                
                # Create key for this catalyst type
                key = f"{catalyst.ticker}_{catalyst.category.value}"
                
                # Check if we've alerted for this catalyst recently
                if key in self._alert_history:
                    last_alert_time = self._alert_history[key]['last_alert']
                    if current_time - last_alert_time < rate_limit_window:
                        return True
                
                # Check hourly limit
                hour_key = current_time.replace(minute=0, second=0, microsecond=0)
                hourly_count = sum(
                    1 for alert_data in self._alert_history.values()
                    if alert_data['last_alert'].replace(minute=0, second=0, microsecond=0) == hour_key
                )
                
                if hourly_count >= self.config.alerts.max_alerts_per_hour:
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def _update_alert_history(self, catalyst: Catalyst):
        """Update alert history for rate limiting"""
        try:
            with self._rate_limit_lock:
                key = f"{catalyst.ticker}_{catalyst.category.value}"
                self._alert_history[key] = {
                    'last_alert': datetime.now(timezone.utc),
                    'count': self._alert_history.get(key, {}).get('count', 0) + 1
                }
                
                # Clean up old history (keep last 24 hours)
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                expired_keys = [
                    k for k, v in self._alert_history.items()
                    if v['last_alert'] < cutoff_time
                ]
                
                for key in expired_keys:
                    del self._alert_history[key]
                    
        except Exception as e:
            logger.error(f"Error updating alert history: {e}")
    
    async def _send_alerts(self, catalyst: Catalyst) -> Dict[str, Any]:
        """Send alerts through all enabled channels"""
        results = {}
        
        # Create alert message
        message = self._format_alert_message(catalyst)
        subject = f"Catalyst Alert: {catalyst.ticker} - {catalyst.category.value}"
        
        # Send through enabled channels
        for channel_name in self.config.alerts.enabled_channels:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    result = await channel.send_alert(
                        message=message,
                        subject=subject,
                        catalyst=catalyst
                    )
                    results[channel_name] = result
                except Exception as e:
                    logger.error(f"Error sending alert via {channel_name}: {e}")
                    results[channel_name] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                logger.warning(f"Alert channel not configured: {channel_name}")
                results[channel_name] = {
                    'success': False,
                    'error': 'Channel not configured'
                }
        
        return results
    
    def _format_alert_message(self, catalyst: Catalyst) -> str:
        """Format catalyst information into alert message"""
        try:
            sentiment_icon = {
                'Positive': '🟢',
                'Negative': '🔴',
                'Neutral': '⚪'
            }.get(catalyst.sentiment_label.value, '⚪')
            
            message = f"""
🚨 CATALYST ALERT 🚨

{sentiment_icon} Ticker: {catalyst.ticker}
📊 Category: {catalyst.category.value}
💪 Impact Score: {catalyst.impact}/100
🎯 Confidence: {catalyst.confidence:.1%}
📈 Sentiment: {catalyst.sentiment_label.value} ({catalyst.sentiment_score:.2f})

📝 Summary:
{catalyst.catalyst}

🏢 Sector: {catalyst.sector}
🔗 Source: {catalyst.source.value}
⏰ Time: {catalyst.published_date.strftime('%Y-%m-%d %H:%M UTC') if catalyst.published_date else 'N/A'}

{f'🌐 URL: {catalyst.url}' if catalyst.url else ''}
""".strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return f"Catalyst Alert: {catalyst.ticker} - {catalyst.category.value}"
    
    async def send_test_alert(self, channel_name: str, recipient: str = None) -> Dict[str, Any]:
        """Send a test alert through specified channel"""
        try:
            if channel_name not in self.channels:
                return {
                    'success': False,
                    'error': f'Channel {channel_name} not configured'
                }
            
            # Create test catalyst
            test_catalyst = Catalyst(
                ticker='TEST',
                catalyst='This is a test alert from the Financial Catalyst Scanner',
                category='General',
                sentiment_label='Positive',
                sentiment_score=0.5,
                impact=75,
                source='Manual',
                confidence=0.9,
                published_date=datetime.now(timezone.utc)
            )
            
            channel = self.channels[channel_name]
            message = self._format_alert_message(test_catalyst)
            subject = "Test Alert - Financial Catalyst Scanner"
            
            result = await channel.send_alert(
                message=message,
                subject=subject,
                catalyst=test_catalyst,
                recipient=recipient
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending test alert: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics"""
        return {
            **self._stats,
            'configured_channels': list(self.channels.keys()),
            'enabled_channels': self.config.alerts.enabled_channels,
            'alert_rules_count': len(self.alert_rules),
            'rate_limit_entries': len(self._alert_history)
        }
    
    def get_channel_status(self) -> Dict[str, Any]:
        """Get status of all notification channels"""
        status = {}
        for name, channel in self.channels.items():
            try:
                channel_status = channel.get_status()
                status[name] = channel_status
            except Exception as e:
                status[name] = {
                    'configured': False,
                    'error': str(e)
                }
        return status
    
    async def process_trading_opportunity(self, opportunity: TradingOpportunity) -> Dict[str, Any]:
        """Process a trading opportunity and send alerts if criteria are met"""
        try:
            # Check if opportunity meets alert criteria
            if not self._should_alert_trading_opportunity(opportunity):
                return {
                    'alert_sent': False,
                    'reason': 'filtered',
                    'ticker': opportunity.ticker
                }
            
            # Check rate limiting
            if await self._is_trading_opportunity_rate_limited(opportunity):
                return {
                    'alert_sent': False,
                    'reason': 'rate_limited',
                    'ticker': opportunity.ticker
                }
            
            # Send trading signal alerts
            results = await self._send_trading_signal_alerts(opportunity)
            
            if any(result.get('success', False) for result in results.values()):
                self._stats['alerts_sent'] += 1
                self._stats['last_alert'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Trading signal alert sent: {opportunity.ticker} - {opportunity.trading_type}")
                
                # Update rate limiting history
                await self._update_trading_opportunity_history(opportunity)
                
                return {
                    'alert_sent': True,
                    'channels': list(results.keys()),
                    'ticker': opportunity.ticker,
                    'results': results
                }
            else:
                self._stats['alerts_failed'] += 1
                return {
                    'alert_sent': False,
                    'reason': 'send_failed',
                    'ticker': opportunity.ticker,
                    'results': results
                }
                
        except Exception as e:
            logger.error(f"Error processing trading opportunity alert: {e}")
            self._stats['alerts_failed'] += 1
            return {
                'alert_sent': False,
                'reason': 'error',
                'error': str(e),
                'ticker': opportunity.ticker if opportunity else 'unknown'
            }
    
    def _should_alert_trading_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """Check if trading opportunity should trigger an alert"""
        try:
            # Minimum opportunity score threshold
            if opportunity.opportunity_score < 0.7:
                return False
            
            # Minimum confidence threshold
            if opportunity.confidence < 0.75:
                return False
            
            # High-confidence, high-impact opportunities only
            if opportunity.opportunity_score >= 0.85 and opportunity.confidence >= 0.8:
                return True
            
            # Strong signals with good risk/reward
            if (opportunity.signal_strength >= 0.8 and 
                opportunity.risk_reward_ratio and 
                opportunity.risk_reward_ratio >= 2.0):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trading opportunity alert criteria: {e}")
            return False
    
    async def _is_trading_opportunity_rate_limited(self, opportunity: TradingOpportunity) -> bool:
        """Check if trading opportunity is rate limited"""
        try:
            with self._rate_limit_lock:
                key = f"trading_{opportunity.ticker}_{opportunity.trading_type}"
                now = datetime.now(timezone.utc)
                
                if key in self._alert_history:
                    last_alert = self._alert_history[key]
                    time_diff = now - last_alert
                    
                    # Rate limit: max 1 trading signal per ticker per trading type per hour
                    if time_diff < timedelta(hours=1):
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking trading opportunity rate limit: {e}")
            return False
    
    async def _send_trading_signal_alerts(self, opportunity: TradingOpportunity) -> Dict[str, Dict]:
        """Send trading signal alerts through all configured channels"""
        results = {}
        message = self._format_trading_signal_message(opportunity)
        subject = f"🚀 AI Trading Signal: {opportunity.ticker} - {opportunity.trading_type.upper()}"
        
        # Send through enabled channels
        for channel_name in self.config.alerts.enabled_channels:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    result = await channel.send_alert(
                        message=message,
                        subject=subject,
                        catalyst=None,  # Not a catalyst alert
                        trading_opportunity=opportunity
                    )
                    results[channel_name] = result
                    
                except Exception as e:
                    logger.error(f"Error sending alert via {channel_name}: {e}")
                    results[channel_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    def _format_trading_signal_message(self, opportunity: TradingOpportunity) -> str:
        """Format trading signal alert message"""
        try:
            direction_emoji = "🟢" if opportunity.direction == 1 else "🔴" if opportunity.direction == -1 else "🟡"
            action = "BUY" if opportunity.direction == 1 else "SELL" if opportunity.direction == -1 else "WATCH"
            
            message = f"""
🚀 **AI TRADING SIGNAL ALERT**

**{direction_emoji} {action} Signal: {opportunity.ticker}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **Signal Details:**
• Opportunity Score: {opportunity.opportunity_score:.1%}
• Signal Strength: {opportunity.signal_strength:.1%}
• Confidence: {opportunity.confidence:.1%}
• Trading Type: {opportunity.trading_type.replace('_', ' ').title()}
• Risk Level: {opportunity.risk_level}

💰 **Price Targets:**
• Entry: ${opportunity.entry_price:.2f}
• Target: ${opportunity.target_price:.2f}
• Stop Loss: ${opportunity.stop_loss_price:.2f}
• Est. Return: {opportunity.estimated_return:.1%}
• Risk/Reward: {opportunity.risk_reward_ratio:.1f}:1

📈 **Market Context:**
• Sector: {opportunity.sector}
• Market Cap: {opportunity.market_cap_category}
• Volatility: {opportunity.volatility_level}
• Time Horizon: {opportunity.time_horizon}

🔍 **Key Factors:**
{chr(10).join(f"• {factor}" for factor in opportunity.key_factors[:3])}

⚠️ **Risk Management:**
Risk Level: {opportunity.risk_level}
Max suggested position size: Calculate based on your risk tolerance

📝 **Disclaimer:** This is an AI-generated trading signal for educational purposes. Always conduct your own research and consider your risk tolerance before trading.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
            """
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting trading signal message: {e}")
            return f"Trading Signal Alert: {opportunity.ticker} - {action} Signal"
    
    async def _update_trading_opportunity_history(self, opportunity: TradingOpportunity):
        """Update trading opportunity alert history for rate limiting"""
        try:
            with self._rate_limit_lock:
                key = f"trading_{opportunity.ticker}_{opportunity.trading_type}"
                self._alert_history[key] = datetime.now(timezone.utc)
                
                # Cleanup old entries (keep last 24 hours)
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                self._alert_history = {
                    k: v for k, v in self._alert_history.items() 
                    if v > cutoff
                }
                
        except Exception as e:
            logger.error(f"Error updating trading opportunity history: {e}")
    
    async def cleanup(self):
        """Cleanup alert manager resources"""
        try:
            # Cleanup all channels
            for channel in self.channels.values():
                if hasattr(channel, 'cleanup'):
                    await channel.cleanup()
            
            logger.info("Alert Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during alert manager cleanup: {e}")
