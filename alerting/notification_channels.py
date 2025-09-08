import logging
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import aiohttp
import json

from shared.models import Catalyst

logger = logging.getLogger(__name__)

class BaseNotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self._stats = {
            'sent': 0,
            'failed': 0,
            'last_sent': None
        }
    
    async def send_alert(self, message: str, subject: str, catalyst: Catalyst, recipient: str = None) -> Dict[str, Any]:
        """Send alert through this channel"""
        raise NotImplementedError("Subclasses must implement send_alert method")
    
    def get_status(self) -> Dict[str, Any]:
        """Get channel status"""
        return {
            'name': self.name,
            'configured': True,
            'stats': self._stats
        }
    
    def _update_stats(self, success: bool):
        """Update channel statistics"""
        if success:
            self._stats['sent'] += 1
            self._stats['last_sent'] = datetime.now(timezone.utc).isoformat()
        else:
            self._stats['failed'] += 1

class ConsoleChannel(BaseNotificationChannel):
    """Console/logging notification channel"""
    
    def __init__(self):
        super().__init__()
    
    async def send_alert(self, message: str, subject: str, catalyst: Catalyst, recipient: str = None) -> Dict[str, Any]:
        """Send alert to console/logs"""
        try:
            separator = "=" * 60
            
            alert_output = f"""
{separator}
{subject}
{separator}
{message}
{separator}
""".strip()
            
            # Log at INFO level for alerts
            logger.info(f"\n{alert_output}")
            
            # Also print to console for immediate visibility
            print(f"\n{alert_output}")
            
            self._update_stats(True)
            
            return {
                'success': True,
                'channel': 'console',
                'message': 'Alert logged to console'
            }
            
        except Exception as e:
            logger.error(f"Console alert error: {e}")
            self._update_stats(False)
            return {
                'success': False,
                'channel': 'console',
                'error': str(e)
            }

class EmailChannel(BaseNotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_alert(self, message: str, subject: str, catalyst: Catalyst, recipient: str = None) -> Dict[str, Any]:
        """Send alert via email"""
        try:
            if not recipient:
                # Use default recipient (could be configured)
                recipient = self.username  # Send to sender as default
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add HTML formatting
            html_message = self._format_html_message(message, catalyst)
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email in thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email, msg, recipient
            )
            
            self._update_stats(True)
            
            return {
                'success': True,
                'channel': 'email',
                'recipient': recipient,
                'message': 'Email sent successfully'
            }
            
        except Exception as e:
            logger.error(f"Email alert error: {e}")
            self._update_stats(False)
            return {
                'success': False,
                'channel': 'email',
                'error': str(e)
            }
    
    def _send_email(self, msg: MIMEMultipart, recipient: str):
        """Send email synchronously"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            raise
    
    def _format_html_message(self, message: str, catalyst: Catalyst) -> str:
        """Format message as HTML email"""
        try:
            # Convert plain text message to HTML with basic formatting
            html_message = message.replace('\n', '<br>')
            
            # Add some styling
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                            <h1 style="margin: 0; font-size: 24px;">🚨 Catalyst Alert</h1>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff;">
                            <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; margin: 0;">
{html_message}
                            </pre>
                        </div>
                        
                        <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; font-size: 12px; color: #666;">
                            <p style="margin: 0;"><strong>Generated by:</strong> Financial Catalyst Scanner</p>
                            <p style="margin: 5px 0 0 0;"><strong>Time:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error formatting HTML email: {e}")
            return f"<html><body><pre>{message}</pre></body></html>"
    
    def get_status(self) -> Dict[str, Any]:
        """Get email channel status"""
        return {
            'name': self.name,
            'configured': bool(self.smtp_server and self.username and self.password),
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'username': self.username,
            'stats': self._stats
        }

class SMSChannel(BaseNotificationChannel):
    """SMS notification channel using Twilio"""
    
    def __init__(self, account_sid: str, auth_token: str, from_phone: str):
        super().__init__()
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_phone = from_phone
        
        # Initialize Twilio client
        try:
            from twilio.rest import Client
            self.client = Client(account_sid, auth_token)
        except ImportError:
            logger.error("Twilio package not installed. SMS alerts will not work.")
            self.client = None
    
    async def send_alert(self, message: str, subject: str, catalyst: Catalyst, recipient: str = None) -> Dict[str, Any]:
        """Send alert via SMS"""
        try:
            if not self.client:
                return {
                    'success': False,
                    'channel': 'sms',
                    'error': 'Twilio client not available'
                }
            
            if not recipient:
                return {
                    'success': False,
                    'channel': 'sms',
                    'error': 'SMS recipient phone number required'
                }
            
            # Format message for SMS (shorter)
            sms_message = self._format_sms_message(catalyst)
            
            # Send SMS in thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_sms, sms_message, recipient
            )
            
            self._update_stats(True)
            
            return {
                'success': True,
                'channel': 'sms',
                'recipient': recipient,
                'message': 'SMS sent successfully'
            }
            
        except Exception as e:
            logger.error(f"SMS alert error: {e}")
            self._update_stats(False)
            return {
                'success': False,
                'channel': 'sms',
                'error': str(e)
            }
    
    def _send_sms(self, message: str, recipient: str):
        """Send SMS synchronously"""
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=recipient
            )
            logger.info(f"SMS sent with SID: {message.sid}")
            
        except Exception as e:
            logger.error(f"Twilio SMS error: {e}")
            raise
    
    def _format_sms_message(self, catalyst: Catalyst) -> str:
        """Format catalyst for SMS (keep it short)"""
        try:
            sentiment_icon = {
                'Positive': '📈',
                'Negative': '📉',
                'Neutral': '➡️'
            }.get(catalyst.sentiment_label.value, '➡️')
            
            # Keep SMS under 160 characters if possible
            message = f"🚨 {catalyst.ticker} {sentiment_icon}\n{catalyst.category.value}\nImpact: {catalyst.impact}/100\n{catalyst.catalyst[:60]}{'...' if len(catalyst.catalyst) > 60 else ''}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting SMS: {e}")
            return f"Catalyst Alert: {catalyst.ticker} - {catalyst.category.value}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get SMS channel status"""
        return {
            'name': self.name,
            'configured': bool(self.client and self.account_sid and self.from_phone),
            'from_phone': self.from_phone,
            'twilio_configured': bool(self.client),
            'stats': self._stats
        }

class WebhookChannel(BaseNotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self):
        super().__init__()
    
    async def send_alert(self, message: str, subject: str, catalyst: Catalyst, recipient: str = None) -> Dict[str, Any]:
        """Send alert via webhook"""
        try:
            if not recipient:  # recipient should be webhook URL
                return {
                    'success': False,
                    'channel': 'webhook',
                    'error': 'Webhook URL required as recipient'
                }
            
            # Create webhook payload
            payload = {
                'alert_type': 'catalyst',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'subject': subject,
                'message': message,
                'catalyst': {
                    'ticker': catalyst.ticker,
                    'category': catalyst.category.value,
                    'sentiment_label': catalyst.sentiment_label.value,
                    'sentiment_score': catalyst.sentiment_score,
                    'impact': catalyst.impact,
                    'confidence': catalyst.confidence,
                    'source': catalyst.source.value,
                    'url': catalyst.url,
                    'published_date': catalyst.published_date.isoformat() if catalyst.published_date else None
                }
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    recipient,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in [200, 201, 202]:
                        self._update_stats(True)
                        return {
                            'success': True,
                            'channel': 'webhook',
                            'webhook_url': recipient,
                            'response_status': response.status,
                            'message': 'Webhook sent successfully'
                        }
                    else:
                        error_text = await response.text()
                        self._update_stats(False)
                        return {
                            'success': False,
                            'channel': 'webhook',
                            'error': f'HTTP {response.status}: {error_text}'
                        }
            
        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
            self._update_stats(False)
            return {
                'success': False,
                'channel': 'webhook',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get webhook channel status"""
        return {
            'name': self.name,
            'configured': True,  # Webhook is always "configured" but needs URL at send time
            'stats': self._stats
        }
