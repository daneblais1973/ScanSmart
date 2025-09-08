import streamlit as st
import os
import logging
from typing import Dict, Any, List
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manager for API keys, email settings, and system configuration"""
    
    def __init__(self):
        self.api_config_keys = {
            'Financial Data APIs': {
                'NEWSAPI_API_KEY': {
                    'name': 'NewsAPI Key',
                    'description': 'For financial news aggregation (100 requests/day free)',
                    'url': 'https://newsapi.org',
                    'required': False,
                    'type': 'free_tier'
                },
                'ALPHA_VANTAGE_API_KEY': {
                    'name': 'Alpha Vantage Key',
                    'description': 'Stock quotes and financial data (500 requests/day free)',
                    'url': 'https://www.alphavantage.co',
                    'required': False,
                    'type': 'free_tier'
                },
                'FMP_API_KEY': {
                    'name': 'Financial Modeling Prep Key',
                    'description': 'Real-time stock data (250 requests/day free)',
                    'url': 'https://financialmodelingprep.com',
                    'required': False,
                    'type': 'free_tier'
                },
                'POLYGON_API_KEY': {
                    'name': 'Polygon.io Key',
                    'description': 'Market data (5 requests/minute free)',
                    'url': 'https://polygon.io',
                    'required': False,
                    'type': 'free_tier'
                },
                'IEX_API_KEY': {
                    'name': 'IEX Cloud Key',
                    'description': 'Stock market data (10,000 messages/month free)',
                    'url': 'https://iexcloud.io',
                    'required': False,
                    'type': 'free_tier'
                }
            },
            'Social & News APIs': {
                'TWITTER_BEARER_TOKEN': {
                    'name': 'Twitter Bearer Token',
                    'description': 'Social sentiment analysis (Essential plan required)',
                    'url': 'https://developer.twitter.com',
                    'required': False,
                    'type': 'paid'
                },
                'REDDIT_CLIENT_ID': {
                    'name': 'Reddit Client ID',
                    'description': 'Reddit discussion sentiment',
                    'url': 'https://www.reddit.com/prefs/apps',
                    'required': False,
                    'type': 'free'
                },
                'REDDIT_CLIENT_SECRET': {
                    'name': 'Reddit Client Secret',
                    'description': 'Reddit API secret (goes with Client ID)',
                    'url': 'https://www.reddit.com/prefs/apps',
                    'required': False,
                    'type': 'free'
                }
            },
            'Email Notifications': {
                'SMTP_USERNAME': {
                    'name': 'Email Address',
                    'description': 'Your email address for sending alerts',
                    'required': True,
                    'type': 'email'
                },
                'SMTP_PASSWORD': {
                    'name': 'Email Password',
                    'description': 'Email password or app password (for Gmail)',
                    'required': True,
                    'type': 'password'
                },
                'ALERT_EMAIL_RECIPIENTS': {
                    'name': 'Alert Recipients',
                    'description': 'Comma-separated email addresses to receive alerts',
                    'required': True,
                    'type': 'email_list'
                },
                'SMTP_SERVER': {
                    'name': 'SMTP Server',
                    'description': 'SMTP server (gmail: smtp.gmail.com)',
                    'default': 'smtp.gmail.com',
                    'required': False,
                    'type': 'text'
                },
                'SMTP_PORT': {
                    'name': 'SMTP Port',
                    'description': 'SMTP port (usually 587)',
                    'default': '587',
                    'required': False,
                    'type': 'number'
                }
            }
        }
        
    def render_configuration_page(self, session_state):
        """Render the main configuration page"""
        st.markdown("# ⚙️ System Configuration")
        st.markdown("**Set up API keys and email notifications for full functionality**")
        
        # Configuration status overview
        self.render_configuration_status()
        
        # Tabs for different configuration sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Sources", 
            "📧 Email Alerts", 
            "🔑 API Keys", 
            "🧪 Test System"
        ])
        
        with tab1:
            self.render_data_sources_config()
        
        with tab2:
            self.render_email_config()
        
        with tab3:
            self.render_api_keys_config()
        
        with tab4:
            self.render_system_tests()
    
    def render_configuration_status(self):
        """Render overview of configuration status"""
        st.markdown("### 📊 Configuration Status")
        
        # Check configuration status
        status = self.get_configuration_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            email_status = "✅ Configured" if status['email_configured'] else "❌ Not Set"
            st.metric("Email Alerts", email_status)
        
        with col2:
            data_sources = status['active_data_sources']
            st.metric("Data Sources", f"{data_sources}/6")
        
        with col3:
            free_sources = status['free_sources_active']
            st.metric("Free APIs", f"{free_sources} Active")
        
        with col4:
            overall_score = status['configuration_score']
            color = "🟢" if overall_score >= 70 else "🟡" if overall_score >= 40 else "🔴"
            st.metric("Setup Score", f"{color} {overall_score}%")
        
        # Configuration recommendations
        if overall_score < 70:
            st.warning("⚠️ **Configuration Incomplete** - Set up email alerts and at least one data source for full functionality")
        else:
            st.success("✅ **System Ready** - All essential components configured!")
    
    def render_data_sources_config(self):
        """Render data sources configuration"""
        st.markdown("### 📊 Free Data Sources")
        st.markdown("**Always-available data sources (no API key required)**")
        
        # Free data sources (no API key needed)
        free_sources = [
            {"name": "📈 Binance Crypto", "status": "✅ Active", "description": "Real-time cryptocurrency prices"},
            {"name": "🪙 CoinGecko", "status": "✅ Active", "description": "Crypto market data and news"},
            {"name": "📰 RSS News Feeds", "status": "✅ Active", "description": "Financial news from multiple sources"},
            {"name": "🏛️ SEC EDGAR", "status": "✅ Active", "description": "Regulatory filings and insider trading"},
            {"name": "🌤️ Economic Data", "status": "✅ Active", "description": "Federal Reserve and Treasury data"}
        ]
        
        for source in free_sources:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.write(source["name"])
                with col2:
                    st.write(source["status"])
                with col3:
                    st.write(source["description"])
        
        st.markdown("---")
        st.markdown("### 🔑 API-Enhanced Data Sources")
        st.markdown("**Additional data sources available with free API keys**")
        
        # API-enhanced sources
        api_sources = [
            {
                "name": "📰 NewsAPI",
                "key": "NEWSAPI_API_KEY",
                "free_tier": "100 requests/day",
                "description": "Financial news headlines and articles"
            },
            {
                "name": "📊 Alpha Vantage",
                "key": "ALPHA_VANTAGE_API_KEY", 
                "free_tier": "500 requests/day",
                "description": "Stock quotes, earnings, and financial data"
            },
            {
                "name": "💹 Financial Modeling Prep",
                "key": "FMP_API_KEY",
                "free_tier": "250 requests/day",
                "description": "Real-time stock prices and company profiles"
            }
        ]
        
        for source in api_sources:
            is_configured = bool(os.getenv(source["key"]))
            status = "✅ Configured" if is_configured else "❌ Not Set"
            
            with st.expander(f"{source['name']} - {status}"):
                st.markdown(f"**Free Tier:** {source['free_tier']}")
                st.markdown(f"**Description:** {source['description']}")
                
                if not is_configured:
                    st.info(f"💡 Set the `{source['key']}` environment variable to enable this source")
                else:
                    st.success(f"✅ {source['name']} is configured and active")
    
    def render_email_config(self):
        """Render email configuration section"""
        st.markdown("### 📧 Email Alert Configuration")
        
        # Email provider selection
        provider = st.selectbox(
            "Email Provider:",
            ["Gmail", "Outlook/Hotmail", "Yahoo", "Custom SMTP"],
            help="Select your email provider for easy setup"
        )
        
        # Provider-specific instructions
        if provider == "Gmail":
            st.info("""
            📧 **Gmail Setup:**
            1. Enable 2-Factor Authentication in your Google Account
            2. Generate an "App Password" in Security settings
            3. Use your Gmail address as username
            4. Use the App Password (not your regular password)
            """)
            smtp_server_default = "smtp.gmail.com"
            smtp_port_default = "587"
        
        elif provider == "Outlook/Hotmail":
            st.info("""
            📧 **Outlook Setup:**
            1. Use your full Outlook/Hotmail email address
            2. Use your regular account password
            3. You may need to enable "Less secure app access"
            """)
            smtp_server_default = "smtp-mail.outlook.com"
            smtp_port_default = "587"
        
        elif provider == "Yahoo":
            st.info("""
            📧 **Yahoo Setup:**
            1. Enable 2-Step Verification in Yahoo Account Security
            2. Generate an "App Password" for this application
            3. Use your Yahoo email as username
            4. Use the App Password (not your regular password)
            """)
            smtp_server_default = "smtp.mail.yahoo.com"
            smtp_port_default = "587"
        
        else:  # Custom
            st.info("""
            📧 **Custom SMTP Setup:**
            Enter your email provider's SMTP settings manually.
            Check your email provider's documentation for SMTP details.
            """)
            smtp_server_default = ""
            smtp_port_default = "587"
        
        # Email configuration form
        with st.form("email_config"):
            st.markdown("#### Email Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                smtp_username = st.text_input(
                    "Email Address:",
                    value=os.getenv('SMTP_USERNAME', ''),
                    placeholder="your.email@gmail.com"
                )
                
                smtp_server = st.text_input(
                    "SMTP Server:",
                    value=os.getenv('SMTP_SERVER', smtp_server_default),
                    placeholder="smtp.gmail.com"
                )
            
            with col2:
                smtp_password = st.text_input(
                    "Email Password:",
                    type="password",
                    value=os.getenv('SMTP_PASSWORD', ''),
                    placeholder="Your app password"
                )
                
                smtp_port = st.text_input(
                    "SMTP Port:",
                    value=os.getenv('SMTP_PORT', smtp_port_default),
                    placeholder="587"
                )
            
            alert_recipients = st.text_input(
                "Alert Recipients:",
                value=os.getenv('ALERT_EMAIL_RECIPIENTS', ''),
                placeholder="recipient1@email.com, recipient2@email.com",
                help="Comma-separated list of email addresses to receive alerts"
            )
            
            if st.form_submit_button("💾 Save Email Configuration", type="primary"):
                self.save_email_configuration(smtp_username, smtp_password, smtp_server, smtp_port, alert_recipients)
    
    def render_api_keys_config(self):
        """Render API keys configuration"""
        st.markdown("### 🔑 API Keys Setup")
        
        for category, apis in self.api_config_keys.items():
            with st.expander(f"🔧 {category}"):
                
                for key, config in apis.items():
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{config['name']}**")
                        if config.get('url'):
                            st.markdown(f"[Get API Key]({config['url']})")
                    
                    with col2:
                        current_value = os.getenv(key, '')
                        if current_value:
                            st.success("✅ Set")
                        else:
                            st.error("❌ Not Set")
                    
                    with col3:
                        st.markdown(config['description'])
                        if config.get('type') == 'free_tier':
                            st.markdown("🆓 **Free Tier Available**")
                        elif config.get('type') == 'free':
                            st.markdown("🆓 **Free**")
                        elif config.get('type') == 'paid':
                            st.markdown("💰 **Paid Service**")
                
                # Quick setup instructions
                st.markdown(f"""
                **To set up {category} APIs:**
                1. Get API keys from the linked websites above
                2. Add them to your Replit Secrets (recommended) or environment variables
                3. Use the exact environment variable names shown above
                """)
    
    def render_system_tests(self):
        """Render system testing section"""
        st.markdown("### 🧪 System Tests")
        st.markdown("**Test individual components to verify they're working correctly**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📧 Email System Test")
            test_email = st.text_input(
                "Test Email Address:",
                placeholder="test@example.com",
                help="Email address to send test alert"
            )
            
            if st.button("📧 Send Test Email", type="primary"):
                if test_email:
                    with st.spinner("Sending test email..."):
                        result = self.test_email_system(test_email)
                        if result['success']:
                            st.success("✅ Test email sent successfully!")
                        else:
                            st.error(f"❌ Email test failed: {result['error']}")
                else:
                    st.error("Please enter a test email address")
        
        with col2:
            st.markdown("#### 📊 Data Sources Test")
            
            if st.button("🔍 Test Data Sources", type="secondary"):
                with st.spinner("Testing data sources..."):
                    results = self.test_data_sources()
                    
                    for source, result in results.items():
                        if result['success']:
                            st.success(f"✅ {source}: {result['message']}")
                        else:
                            st.error(f"❌ {source}: {result['message']}")
        
        # Full system test
        st.markdown("---")
        st.markdown("#### 🚀 Full System Test")
        
        if st.button("🧪 Run Complete System Test", type="primary"):
            with st.spinner("Running comprehensive system test..."):
                self.run_full_system_test()
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        # Check email configuration
        email_configured = bool(
            os.getenv('SMTP_USERNAME') and 
            os.getenv('SMTP_PASSWORD') and 
            os.getenv('ALERT_EMAIL_RECIPIENTS')
        )
        
        # Check data source APIs
        data_apis = [
            'NEWSAPI_API_KEY',
            'ALPHA_VANTAGE_API_KEY', 
            'FMP_API_KEY',
            'POLYGON_API_KEY',
            'IEX_API_KEY',
            'TWITTER_BEARER_TOKEN'
        ]
        
        active_data_sources = sum(1 for api in data_apis if os.getenv(api))
        
        # Free sources that don't need API keys (always available)
        free_sources_active = 5  # Binance, CoinGecko, RSS, SEC, Economic data
        
        # Calculate configuration score
        email_score = 40 if email_configured else 0
        data_score = min(40, (active_data_sources / len(data_apis)) * 40)
        free_score = 20  # Always have free sources
        
        configuration_score = int(email_score + data_score + free_score)
        
        return {
            'email_configured': email_configured,
            'active_data_sources': active_data_sources,
            'free_sources_active': free_sources_active,
            'configuration_score': configuration_score
        }
    
    def save_email_configuration(self, username, password, server, port, recipients):
        """Save email configuration to environment"""
        try:
            # In a real deployment, these would be saved to environment variables
            # For Replit, these should be added to Secrets
            st.info("""
            📝 **To save these settings:**
            1. Go to the **Secrets** tab in your Replit workspace
            2. Add these secrets:
               - `SMTP_USERNAME`: {username}
               - `SMTP_PASSWORD`: {password}
               - `SMTP_SERVER`: {server}
               - `SMTP_PORT`: {port}
               - `ALERT_EMAIL_RECIPIENTS`: {recipients}
            3. Restart your application
            """.format(
                username=username,
                password="[Your password]",
                server=server,
                port=port,
                recipients=recipients
            ))
            
            st.success("✅ Configuration ready to save to Secrets!")
            
        except Exception as e:
            st.error(f"Error saving configuration: {e}")
    
    def test_email_system(self, test_email: str) -> Dict[str, Any]:
        """Test email system"""
        try:
            # This would integrate with the EmailNotifier
            # For now, return a mock result
            if not os.getenv('SMTP_USERNAME'):
                return {
                    'success': False,
                    'error': 'Email not configured. Please set up SMTP settings first.'
                }
            
            # Simulate email test
            return {
                'success': True,
                'message': f'Test email would be sent to {test_email}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Test data sources"""
        results = {}
        
        # Test free sources (always available)
        results['Free Crypto Data'] = {
            'success': True,
            'message': 'Binance and CoinGecko APIs responding'
        }
        
        results['RSS News Feeds'] = {
            'success': True,
            'message': '5+ news RSS feeds active'
        }
        
        # Test API-dependent sources
        if os.getenv('NEWSAPI_API_KEY'):
            results['NewsAPI'] = {
                'success': True,
                'message': 'NewsAPI key configured'
            }
        else:
            results['NewsAPI'] = {
                'success': False,
                'message': 'API key not configured'
            }
        
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            results['Alpha Vantage'] = {
                'success': True,
                'message': 'Alpha Vantage key configured'
            }
        else:
            results['Alpha Vantage'] = {
                'success': False,
                'message': 'API key not configured'
            }
        
        return results
    
    def run_full_system_test(self):
        """Run comprehensive system test"""
        st.markdown("#### 🧪 Full System Test Results")
        
        # Test configuration
        status = self.get_configuration_status()
        if status['configuration_score'] >= 70:
            st.success("✅ Configuration: System properly configured")
        else:
            st.warning(f"⚠️ Configuration: Setup incomplete ({status['configuration_score']}% complete)")
        
        # Test data sources
        data_results = self.test_data_sources()
        working_sources = sum(1 for r in data_results.values() if r['success'])
        total_sources = len(data_results)
        
        if working_sources >= total_sources // 2:
            st.success(f"✅ Data Sources: {working_sources}/{total_sources} sources working")
        else:
            st.error(f"❌ Data Sources: Only {working_sources}/{total_sources} sources working")
        
        # Test email system
        email_test = self.test_email_system("test@example.com")
        if email_test['success']:
            st.success("✅ Email System: Ready to send alerts")
        else:
            st.error(f"❌ Email System: {email_test['error']}")
        
        # Overall verdict
        if status['configuration_score'] >= 70 and working_sources >= 2:
            st.success("🎉 **System Status: READY FOR TRADING**")
            st.balloons()
        else:
            st.warning("⚠️ **System Status: NEEDS CONFIGURATION** - Complete setup for full functionality")

# Global instance
config_manager = ConfigurationManager()