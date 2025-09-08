import smtplib
import ssl
import logging
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio
import os

logger = logging.getLogger(__name__)

class EmailNotifier:
    """Professional email notification system for catalyst alerts"""
    
    def __init__(self, config):
        self.config = config
        
        # Email configuration with multiple provider support
        self.smtp_configs = {
            'gmail': {
                'server': 'smtp.gmail.com',
                'port': 587,
                'use_tls': True
            },
            'outlook': {
                'server': 'smtp-mail.outlook.com',
                'port': 587,
                'use_tls': True
            },
            'yahoo': {
                'server': 'smtp.mail.yahoo.com',
                'port': 587,
                'use_tls': True
            },
            'custom': {
                'server': getattr(config.api, 'smtp_server', ''),
                'port': getattr(config.api, 'smtp_port', 587),
                'use_tls': True
            }
        }
        
        # Get email configuration from environment
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        
        # Notification preferences
        self.default_recipients = os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',') if os.getenv('ALERT_EMAIL_RECIPIENTS') else []
        
        logger.info(f"Email Notifier initialized - SMTP: {self.smtp_server}:{self.smtp_port}")
    
    def is_configured(self) -> bool:
        """Check if email notifications are properly configured"""
        return bool(self.smtp_username and self.smtp_password and self.from_email)
    
    async def send_catalyst_alert(self, catalyst: Dict[str, Any], recipients: List[str] = None) -> Dict[str, Any]:
        """Send catalyst alert email"""
        try:
            if not self.is_configured():
                return {
                    'success': False,
                    'error': 'Email not configured. Set SMTP_USERNAME, SMTP_PASSWORD, and SMTP_SERVER environment variables.'
                }
            
            if not recipients:
                recipients = self.default_recipients
            
            if not recipients:
                return {
                    'success': False,
                    'error': 'No email recipients configured. Set ALERT_EMAIL_RECIPIENTS environment variable.'
                }
            
            # Generate email content
            subject = self._generate_subject(catalyst)
            html_content = self._generate_html_content(catalyst)
            text_content = self._generate_text_content(catalyst)
            
            # Send email
            result = await self._send_email(
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            if result['success']:
                logger.info(f"Catalyst alert sent to {len(recipients)} recipients: {catalyst.get('ticker', 'Unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending catalyst alert: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_subject(self, catalyst: Dict[str, Any]) -> str:
        """Generate email subject line"""
        ticker = catalyst.get('ticker', 'Unknown')
        category = catalyst.get('category', 'General')
        impact = catalyst.get('impact', 0)
        
        # Determine urgency emoji
        if impact >= 90:
            urgency = "🚨 CRITICAL"
        elif impact >= 80:
            urgency = "⚡ HIGH"
        elif impact >= 70:
            urgency = "📈 MEDIUM"
        else:
            urgency = "📊 LOW"
        
        return f"{urgency} CATALYST ALERT: {ticker} - {category.title()} Event"
    
    def _generate_html_content(self, catalyst: Dict[str, Any]) -> str:
        """Generate HTML email content"""
        ticker = catalyst.get('ticker', 'Unknown')
        category = catalyst.get('category', 'general')
        impact = catalyst.get('impact', 0)
        sentiment = catalyst.get('sentiment_label', 'neutral')
        confidence = catalyst.get('confidence', 0)
        catalyst_text = catalyst.get('catalyst', 'No details available')
        source = catalyst.get('source', 'Unknown')
        published_date = catalyst.get('published_date', datetime.now(timezone.utc))
        
        # Format published date
        if isinstance(published_date, str):
            try:
                published_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except:
                published_date = datetime.now(timezone.utc)
        
        date_str = published_date.strftime('%Y-%m-%d %H:%M UTC')
        
        # Determine colors based on sentiment
        sentiment_colors = {
            'positive': '#16a34a',  # Green
            'negative': '#dc2626',  # Red
            'neutral': '#6b7280'    # Gray
        }
        
        sentiment_color = sentiment_colors.get(str(sentiment).lower(), '#6b7280')
        
        # Impact level styling
        if impact >= 90:
            impact_color = '#dc2626'  # Red
            impact_text = "CRITICAL"
        elif impact >= 80:
            impact_color = '#ea580c'  # Orange
            impact_text = "HIGH"
        elif impact >= 70:
            impact_color = '#d97706'  # Yellow
            impact_text = "MEDIUM"
        else:
            impact_color = '#059669'  # Teal
            impact_text = "LOW"
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Catalyst Alert - {ticker}</title>
        </head>
        <body style="font-family: Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                
                <!-- Header -->
                <div style="background-color: #1f2937; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">🎯 QuantumCatalyst Pro Alert</h1>
                    <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">Professional Catalyst Detection System</p>
                </div>
                
                <!-- Main Content -->
                <div style="padding: 30px;">
                    
                    <!-- Catalyst Summary -->
                    <div style="background-color: #f9fafb; border-left: 4px solid {sentiment_color}; padding: 20px; margin-bottom: 20px; border-radius: 0 4px 4px 0;">
                        <h2 style="margin: 0 0 10px 0; color: #1f2937; font-size: 20px;">{ticker} - {category.title()} Catalyst</h2>
                        <p style="margin: 0; color: #4b5563; font-size: 16px; line-height: 1.5;">{catalyst_text}</p>
                    </div>
                    
                    <!-- Metrics Grid -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">
                        
                        <!-- Impact Score -->
                        <div style="background-color: #fef3f2; border: 1px solid #fecaca; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">IMPACT SCORE</div>
                            <div style="font-size: 24px; font-weight: bold; color: {impact_color};">{impact}/100</div>
                            <div style="font-size: 12px; color: {impact_color}; font-weight: 500;">{impact_text} PRIORITY</div>
                        </div>
                        
                        <!-- Confidence -->
                        <div style="background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">CONFIDENCE</div>
                            <div style="font-size: 24px; font-weight: bold; color: #0369a1;">{confidence:.1%}</div>
                            <div style="font-size: 12px; color: #0369a1; font-weight: 500;">ANALYSIS CONFIDENCE</div>
                        </div>
                        
                        <!-- Sentiment -->
                        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">SENTIMENT</div>
                            <div style="font-size: 18px; font-weight: bold; color: {sentiment_color};">{str(sentiment).upper()}</div>
                            <div style="font-size: 12px; color: {sentiment_color}; font-weight: 500;">MARKET SENTIMENT</div>
                        </div>
                        
                        <!-- Source -->
                        <div style="background-color: #fefce8; border: 1px solid #fde68a; padding: 15px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">SOURCE</div>
                            <div style="font-size: 16px; font-weight: bold; color: #92400e;">{source}</div>
                            <div style="font-size: 12px; color: #92400e; font-weight: 500;">DATA SOURCE</div>
                        </div>
                        
                    </div>
                    
                    <!-- Timestamp -->
                    <div style="background-color: #f3f4f6; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                        <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">📅 PUBLISHED</div>
                        <div style="font-size: 14px; color: #374151; font-weight: 500;">{date_str}</div>
                    </div>
                    
                    <!-- Action Recommendations -->
                    <div style="background-color: #1f2937; color: white; padding: 20px; border-radius: 6px; margin-bottom: 20px;">
                        <h3 style="margin: 0 0 15px 0; font-size: 16px;">📋 Recommended Actions:</h3>
                        <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                            <li>Review detailed catalyst information and market context</li>
                            <li>Check current stock price and trading volume</li>
                            <li>Analyze sector trends and peer company performance</li>
                            <li>Consider risk management and position sizing</li>
                            <li>Monitor for follow-up news and market reactions</li>
                        </ul>
                    </div>
                    
                </div>
                
                <!-- Footer -->
                <div style="background-color: #f9fafb; padding: 20px; border-radius: 0 0 8px 8px; text-align: center; border-top: 1px solid #e5e7eb;">
                    <p style="margin: 0; font-size: 12px; color: #6b7280;">
                        This alert was generated by QuantumCatalyst Pro - Professional Stock Catalyst Detection System<br>
                        Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </p>
                </div>
                
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_text_content(self, catalyst: Dict[str, Any]) -> str:
        """Generate plain text email content"""
        ticker = catalyst.get('ticker', 'Unknown')
        category = catalyst.get('category', 'general')
        impact = catalyst.get('impact', 0)
        sentiment = catalyst.get('sentiment_label', 'neutral')
        confidence = catalyst.get('confidence', 0)
        catalyst_text = catalyst.get('catalyst', 'No details available')
        source = catalyst.get('source', 'Unknown')
        published_date = catalyst.get('published_date', datetime.now(timezone.utc))
        
        # Format published date
        if isinstance(published_date, str):
            try:
                published_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except:
                published_date = datetime.now(timezone.utc)
        
        date_str = published_date.strftime('%Y-%m-%d %H:%M UTC')
        
        text_content = f"""
🎯 QUANTUMCATALYST PRO ALERT

CATALYST DETECTED: {ticker}
Category: {category.title()}
Impact Score: {impact}/100
Confidence: {confidence:.1%}
Sentiment: {str(sentiment).upper()}
Source: {source}
Published: {date_str}

CATALYST DETAILS:
{catalyst_text}

RECOMMENDED ACTIONS:
• Review detailed catalyst information and market context
• Check current stock price and trading volume  
• Analyze sector trends and peer company performance
• Consider risk management and position sizing
• Monitor for follow-up news and market reactions

---
Generated by QuantumCatalyst Pro at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Professional Stock Catalyst Detection System
        """
        
        return text_content.strip()
    
    async def _send_email(self, recipients: List[str], subject: str, 
                         html_content: str, text_content: str) -> Dict[str, Any]:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            
            # Add plain text and HTML parts
            text_part = MimeText(text_content, 'plain')
            html_part = MimeText(html_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._send_smtp_email, msg, recipients)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating email: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _send_smtp_email(self, msg: MimeMultipart, recipients: List[str]) -> Dict[str, Any]:
        """Send email via SMTP (runs in thread)"""
        try:
            # Create secure connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                
                # Send email
                server.send_message(msg, to_addrs=recipients)
            
            return {
                'success': True,
                'recipients': recipients,
                'sent_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"SMTP send error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def send_test_email(self, test_recipient: str = None) -> Dict[str, Any]:
        """Send test email to verify configuration"""
        try:
            recipients = [test_recipient] if test_recipient else self.default_recipients
            
            if not recipients:
                return {
                    'success': False,
                    'error': 'No test recipient provided'
                }
            
            # Create test catalyst data
            test_catalyst = {
                'ticker': 'TEST',
                'category': 'system_test',
                'impact': 75,
                'sentiment_label': 'positive',
                'confidence': 0.95,
                'catalyst': 'This is a test alert to verify that the QuantumCatalyst Pro email notification system is working correctly.',
                'source': 'System Test',
                'published_date': datetime.now(timezone.utc)
            }
            
            result = await self.send_catalyst_alert(test_catalyst, recipients)
            
            if result['success']:
                logger.info(f"Test email sent successfully to {recipients}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending test email: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_configuration_help(self) -> Dict[str, str]:
        """Get help text for email configuration"""
        return {
            'gmail': "For Gmail: Use your email and an 'App Password' (not your regular password). Enable 2FA and generate an app password in your Google Account settings.",
            'outlook': "For Outlook/Hotmail: Use your email and password. You may need to enable 'Less secure app access' in your account settings.",
            'custom': "For custom SMTP: Set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD environment variables.",
            'environment_variables': """
Required Environment Variables:
- SMTP_USERNAME: Your email address
- SMTP_PASSWORD: Your email password (or app password for Gmail)
- SMTP_SERVER: SMTP server (default: smtp.gmail.com)
- SMTP_PORT: SMTP port (default: 587)
- FROM_EMAIL: From email address (default: same as SMTP_USERNAME)
- ALERT_EMAIL_RECIPIENTS: Comma-separated list of alert recipients
            """
        }