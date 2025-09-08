"""
Enhanced alerting system with real-time monitoring, smart notifications, and multi-channel delivery
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

from data_fetchers.realtime_data import RealTimeDataFetcher, RealTimePrice
from analysis.advanced_analytics import AdvancedAnalytics

logger = logging.getLogger(__name__)

class AlertType(Enum):
    PRICE_TARGET = "price_target"
    VOLUME_SPIKE = "volume_spike"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_PATTERN = "technical_pattern"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_SPIKE = "volatility_spike"
    EARNINGS_SURPRISE = "earnings_surprise"
    INSIDER_ACTIVITY = "insider_activity"

class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    DISABLED = "disabled"
    EXPIRED = "expired"

class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    CONSOLE = "console"
    WEBHOOK = "webhook"
    IN_APP = "in_app"

@dataclass
class Alert:
    id: str
    symbol: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    trigger_count: int = 0
    max_triggers: int = 1
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AlertTrigger:
    alert_id: str
    symbol: str
    alert_type: AlertType
    message: str
    data: Dict[str, Any]
    triggered_at: datetime = None
    
    def __post_init__(self):
        if self.triggered_at is None:
            self.triggered_at = datetime.now()

class EnhancedAlertManager:
    """Advanced alert management with real-time monitoring and smart notifications"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.triggered_alerts: List[AlertTrigger] = []
        self.data_fetcher = RealTimeDataFetcher()
        self.analytics = AdvancedAnalytics()
        self.running = False
        self.check_interval = 30  # seconds
        
        # Notification settings
        self.notification_settings = {
            'email': {
                'enabled': False,
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'username': os.getenv('EMAIL_USERNAME'),
                'password': os.getenv('EMAIL_PASSWORD'),
                'to_address': os.getenv('ALERT_EMAIL_TO')
            },
            'sms': {
                'enabled': False,
                'twilio_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                'twilio_token': os.getenv('TWILIO_AUTH_TOKEN'),
                'from_number': os.getenv('TWILIO_PHONE_NUMBER'),
                'to_number': os.getenv('ALERT_PHONE_NUMBER')
            }
        }
        
    def create_alert(self,
                    symbol: str,
                    alert_type: AlertType,
                    conditions: Dict[str, Any],
                    channels: List[NotificationChannel],
                    expires_in_hours: Optional[int] = 24) -> str:
        """Create a new alert"""
        alert_id = f"{symbol}_{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        alert = Alert(
            id=alert_id,
            symbol=symbol.upper(),
            alert_type=alert_type,
            conditions=conditions,
            channels=channels,
            expires_at=expires_at
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Created alert {alert_id} for {symbol}")
        
        return alert_id
    
    def create_price_target_alert(self,
                                symbol: str,
                                target_price: float,
                                direction: str = "above",
                                channels: List[NotificationChannel] = None) -> str:
        """Create a price target alert"""
        if channels is None:
            channels = [NotificationChannel.CONSOLE, NotificationChannel.IN_APP]
            
        conditions = {
            'target_price': target_price,
            'direction': direction.lower()
        }
        
        return self.create_alert(symbol, AlertType.PRICE_TARGET, conditions, channels)
    
    def create_volume_spike_alert(self,
                                symbol: str,
                                volume_threshold_percent: float = 200,
                                channels: List[NotificationChannel] = None) -> str:
        """Create a volume spike alert"""
        if channels is None:
            channels = [NotificationChannel.CONSOLE, NotificationChannel.IN_APP]
            
        conditions = {
            'threshold_percent': volume_threshold_percent
        }
        
        return self.create_alert(symbol, AlertType.VOLUME_SPIKE, conditions, channels)
    
    def create_volatility_alert(self,
                              symbol: str,
                              volatility_threshold: float = 2.0,
                              channels: List[NotificationChannel] = None) -> str:
        """Create a volatility spike alert"""
        if channels is None:
            channels = [NotificationChannel.CONSOLE, NotificationChannel.IN_APP]
            
        conditions = {
            'volatility_multiple': volatility_threshold
        }
        
        return self.create_alert(symbol, AlertType.VOLATILITY_SPIKE, conditions, channels)
    
    def create_sentiment_alert(self,
                             symbol: str,
                             sentiment_threshold: str = "negative",
                             channels: List[NotificationChannel] = None) -> str:
        """Create a news sentiment alert"""
        if channels is None:
            channels = [NotificationChannel.CONSOLE, NotificationChannel.IN_APP]
            
        conditions = {
            'sentiment': sentiment_threshold.lower()
        }
        
        return self.create_alert(symbol, AlertType.NEWS_SENTIMENT, conditions, channels)
    
    async def check_alerts(self):
        """Check all active alerts"""
        for alert in list(self.alerts.values()):
            if alert.status != AlertStatus.ACTIVE:
                continue
                
            # Check if expired
            if alert.expires_at and datetime.now() > alert.expires_at:
                alert.status = AlertStatus.EXPIRED
                continue
                
            try:
                if await self._check_single_alert(alert):
                    await self._trigger_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {e}")
                
            alert.last_checked = datetime.now()
    
    async def _check_single_alert(self, alert: Alert) -> bool:
        """Check if a single alert should trigger"""
        try:
            if alert.alert_type == AlertType.PRICE_TARGET:
                return await self._check_price_target(alert)
            elif alert.alert_type == AlertType.VOLUME_SPIKE:
                return await self._check_volume_spike(alert)
            elif alert.alert_type == AlertType.VOLATILITY_SPIKE:
                return await self._check_volatility_spike(alert)
            elif alert.alert_type == AlertType.NEWS_SENTIMENT:
                return await self._check_sentiment_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking {alert.alert_type} for {alert.symbol}: {e}")
            
        return False
    
    async def _check_price_target(self, alert: Alert) -> bool:
        """Check price target alert"""
        try:
            price_data = self.data_fetcher.get_live_price(alert.symbol)
            if not price_data:
                return False
                
            target_price = alert.conditions['target_price']
            direction = alert.conditions['direction']
            current_price = price_data.price
            
            if direction == "above" and current_price >= target_price:
                return True
            elif direction == "below" and current_price <= target_price:
                return True
                
        except Exception as e:
            logger.error(f"Error checking price target for {alert.symbol}: {e}")
            
        return False
    
    async def _check_volume_spike(self, alert: Alert) -> bool:
        """Check volume spike alert"""
        try:
            price_data = self.data_fetcher.get_live_price(alert.symbol)
            if not price_data or not price_data.volume:
                return False
                
            # Get historical average volume (simplified)
            import yfinance as yf
            ticker = yf.Ticker(alert.symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return False
                
            avg_volume = hist['Volume'].mean()
            current_volume = price_data.volume
            threshold_percent = alert.conditions['threshold_percent']
            
            volume_ratio = (current_volume / avg_volume) * 100
            
            return volume_ratio >= threshold_percent
            
        except Exception as e:
            logger.error(f"Error checking volume spike for {alert.symbol}: {e}")
            
        return False
    
    async def _check_volatility_spike(self, alert: Alert) -> bool:
        """Check volatility spike alert"""
        try:
            volatility_data = self.analytics.volatility_forecasting(alert.symbol, days_ahead=1)
            
            if not volatility_data:
                return False
                
            current_vol = volatility_data['current_volatility']
            long_term_avg = volatility_data['long_term_average']
            threshold_multiple = alert.conditions['volatility_multiple']
            
            return current_vol >= (long_term_avg * threshold_multiple)
            
        except Exception as e:
            logger.error(f"Error checking volatility spike for {alert.symbol}: {e}")
            
        return False
    
    async def _check_sentiment_alert(self, alert: Alert) -> bool:
        """Check news sentiment alert"""
        try:
            sentiment_data = self.data_fetcher.get_news_sentiment(alert.symbol)
            
            if not sentiment_data:
                return False
                
            current_sentiment = sentiment_data['sentiment']
            target_sentiment = alert.conditions['sentiment']
            
            return current_sentiment == target_sentiment
            
        except Exception as e:
            logger.error(f"Error checking sentiment for {alert.symbol}: {e}")
            
        return False
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications"""
        try:
            # Create trigger message
            message = self._create_alert_message(alert)
            
            trigger = AlertTrigger(
                alert_id=alert.id,
                symbol=alert.symbol,
                alert_type=alert.alert_type,
                message=message,
                data=alert.conditions
            )
            
            self.triggered_alerts.append(trigger)
            
            # Send notifications
            for channel in alert.channels:
                await self._send_notification(channel, trigger)
            
            # Update alert status
            alert.trigger_count += 1
            
            if alert.trigger_count >= alert.max_triggers:
                alert.status = AlertStatus.TRIGGERED
                
            logger.info(f"Triggered alert {alert.id}: {message}")
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.id}: {e}")
    
    def _create_alert_message(self, alert: Alert) -> str:
        """Create a human-readable alert message"""
        symbol = alert.symbol
        
        if alert.alert_type == AlertType.PRICE_TARGET:
            target = alert.conditions['target_price']
            direction = alert.conditions['direction']
            return f"🎯 {symbol} price has gone {direction} ${target:.2f}"
            
        elif alert.alert_type == AlertType.VOLUME_SPIKE:
            threshold = alert.conditions['threshold_percent']
            return f"📊 {symbol} volume spike detected! Volume >{threshold}% of average"
            
        elif alert.alert_type == AlertType.VOLATILITY_SPIKE:
            multiple = alert.conditions['volatility_multiple']
            return f"⚡ {symbol} volatility spike! {multiple}x above normal levels"
            
        elif alert.alert_type == AlertType.NEWS_SENTIMENT:
            sentiment = alert.conditions['sentiment']
            return f"📰 {symbol} news sentiment turned {sentiment}"
        
        return f"🚨 Alert triggered for {symbol}"
    
    async def _send_notification(self, channel: NotificationChannel, trigger: AlertTrigger):
        """Send notification through specified channel"""
        try:
            if channel == NotificationChannel.CONSOLE:
                print(f"[ALERT] {trigger.message}")
                
            elif channel == NotificationChannel.EMAIL and self.notification_settings['email']['enabled']:
                await self._send_email_notification(trigger)
                
            elif channel == NotificationChannel.SMS and self.notification_settings['sms']['enabled']:
                await self._send_sms_notification(trigger)
                
            elif channel == NotificationChannel.IN_APP:
                # Store for in-app display
                pass
                
        except Exception as e:
            logger.error(f"Error sending {channel.value} notification: {e}")
    
    async def _send_email_notification(self, trigger: AlertTrigger):
        """Send email notification"""
        try:
            settings = self.notification_settings['email']
            
            if not all([settings['username'], settings['password'], settings['to_address']]):
                logger.warning("Email notification not configured properly")
                return
            
            msg = MIMEMultipart()
            msg['From'] = settings['username']
            msg['To'] = settings['to_address']
            msg['Subject'] = f"Trading Alert: {trigger.symbol}"
            
            body = f"""
            Alert Triggered: {trigger.symbol}
            
            {trigger.message}
            
            Time: {trigger.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
            Alert Type: {trigger.alert_type.value}
            
            Data: {json.dumps(trigger.data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'])
            server.starttls()
            server.login(settings['username'], settings['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {trigger.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_sms_notification(self, trigger: AlertTrigger):
        """Send SMS notification using Twilio"""
        try:
            settings = self.notification_settings['sms']
            
            if not all([settings['twilio_sid'], settings['twilio_token'], 
                       settings['from_number'], settings['to_number']]):
                logger.warning("SMS notification not configured properly")
                return
            
            from twilio.rest import Client
            
            client = Client(settings['twilio_sid'], settings['twilio_token'])
            
            message = client.messages.create(
                body=f"Trading Alert: {trigger.message}",
                from_=settings['from_number'],
                to=settings['to_number']
            )
            
            logger.info(f"SMS alert sent for {trigger.symbol}: {message.sid}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [
            {
                'id': alert.id,
                'symbol': alert.symbol,
                'type': alert.alert_type.value,
                'conditions': alert.conditions,
                'status': alert.status.value,
                'created_at': alert.created_at.isoformat(),
                'last_checked': alert.last_checked.isoformat() if alert.last_checked else None,
                'trigger_count': alert.trigger_count
            }
            for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]
    
    def get_triggered_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent triggered alerts"""
        recent_triggers = sorted(
            self.triggered_alerts,
            key=lambda x: x.triggered_at,
            reverse=True
        )[:limit]
        
        return [
            {
                'symbol': trigger.symbol,
                'type': trigger.alert_type.value,
                'message': trigger.message,
                'triggered_at': trigger.triggered_at.isoformat(),
                'data': trigger.data
            }
            for trigger in recent_triggers
        ]
    
    def disable_alert(self, alert_id: str) -> bool:
        """Disable an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].status = AlertStatus.DISABLED
            logger.info(f"Disabled alert {alert_id}")
            return True
        return False
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Deleted alert {alert_id}")
            return True
        return False
    
    async def start_monitoring(self):
        """Start the alert monitoring loop"""
        self.running = True
        logger.info("Started alert monitoring")
        
        while self.running:
            try:
                await self.check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop the alert monitoring loop"""
        self.running = False
        logger.info("Stopped alert monitoring")
    
    def configure_email_notifications(self,
                                    smtp_server: str,
                                    smtp_port: int,
                                    username: str,
                                    password: str,
                                    to_address: str):
        """Configure email notifications"""
        self.notification_settings['email'] = {
            'enabled': True,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'to_address': to_address
        }
        
    def configure_sms_notifications(self,
                                  twilio_sid: str,
                                  twilio_token: str,
                                  from_number: str,
                                  to_number: str):
        """Configure SMS notifications"""
        self.notification_settings['sms'] = {
            'enabled': True,
            'twilio_sid': twilio_sid,
            'twilio_token': twilio_token,
            'from_number': from_number,
            'to_number': to_number
        }