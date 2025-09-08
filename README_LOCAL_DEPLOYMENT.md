# Financial Catalyst Scanner - Local Windows 11 Deployment

🎯 **Advanced Financial Intelligence Platform** with AI-powered catalyst detection, real-time monitoring, machine learning model training, backtesting framework, and RESTful API integration.

## 🚀 Quick Start (Windows 11)

### Option 1: Automated Installation
1. **Download all files** to a folder (e.g., `C:\CatalystScanner\`)
2. **Double-click** `install_windows.bat`
3. **Wait** for automatic setup and browser to open
4. **Configure API keys** in the Configuration tab

### Option 2: Manual Installation
1. **Install Python 3.9+** from [python.org](https://python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
2. **Open Command Prompt** and navigate to the project folder:
   ```cmd
   cd C:\CatalystScanner
   ```
3. **Run the launcher**:
   ```cmd
   python run_local.py
   ```

## 📋 Complete File List for Download

### Core Application Files
- `app.py` - Main Streamlit application
- `run_local.py` - Windows 11 launcher script
- `install_windows.bat` - Automated Windows installer
- `setup.py` - Python package configuration

### Core Modules
- `core/`
  - `config.py` - Configuration management
  - `database.py` - Database operations
  - `cache.py` - Caching system
  - `__init__.py`

### User Interface
- `ui/`
  - `dashboard.py` - Main dashboard
  - `catalyst_viewer.py` - Catalyst browser
  - `configuration.py` - API key configuration
  - `__init__.py`

### Data Processing
- `data_fetchers/`
  - `base_fetcher.py` - Base fetcher class
  - `newsapi_fetcher.py` - News API integration
  - `twitter_fetcher.py` - Twitter/X integration
  - `reddit_fetcher.py` - Reddit integration
  - `financial_fetcher.py` - Financial data APIs
  - `regulatory_fetcher.py` - Regulatory data
  - `rss_fetcher.py` - RSS aggregation
  - `__init__.py`

### AI/ML Components
- `nlp/`
  - `processor.py` - NLP processing engine
  - `models.py` - ML model management
  - `sentiment_analyzer.py` - Sentiment analysis
  - `__init__.py`

### Advanced Features (NEW)
- `ml_training/`
  - `model_trainer.py` - ML model retraining pipeline
  - `__init__.py`
- `backtesting/`
  - `backtester.py` - Historical validation framework
  - `__init__.py`
- `api/`
  - `endpoints.py` - RESTful API endpoints
  - `server.py` - API server launcher
  - `__init__.py`

### Alerting System
- `alerting/`
  - `alert_manager.py` - Alert coordination
  - `notification_channels.py` - Multi-channel notifications
  - `__init__.py`

### Background Processing
- `batch_processing/`
  - `scheduler.py` - Job scheduling
  - `processor.py` - Batch processing
  - `__init__.py`

### Shared Components
- `shared/`
  - `models.py` - Data models
  - `utils.py` - Utility functions
  - `__init__.py`

## 🎯 Key Features Available

### ✅ Core Intelligence
- **Real-time Financial Catalyst Detection** with AI analysis
- **Multi-Source Data Ingestion** (News, Social, Regulatory, RSS)
- **Advanced NLP Processing** with financial sentiment analysis
- **Smart Impact Scoring** and confidence metrics

### ✅ NEW Advanced Features
- **🤖 ML Model Retraining Pipeline** - Automated model updates with performance monitoring
- **📈 Backtesting Framework** - Historical validation of catalyst detection accuracy  
- **🚀 RESTful API** - External integration endpoints for mobile/third-party applications

### ✅ User Experience
- **📊 Interactive Streamlit Dashboard** - Real-time monitoring and analysis
- **⚙️ UI-Based API Configuration** - Easy setup of all API keys through web interface
- **📱 Multi-Channel Alerting** - SMS, Email, Console, and Webhook notifications
- **🔍 Comprehensive Search & Filtering** - Advanced catalyst exploration

## 🔑 API Keys Configuration

After starting the application, go to the **Configuration** tab to set up:

### Data Sources
- **NewsAPI Key** - Financial news monitoring
- **Twitter Bearer Token** - Social sentiment analysis
- **Reddit API Credentials** - Community discussion tracking

### AI Services  
- **OpenAI API Key** - Advanced text analysis and summarization

### Alerting
- **Twilio Credentials** - SMS alert notifications

## 💻 System Requirements

- **Operating System**: Windows 11 (also compatible with Windows 10, macOS, Linux)
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for data sources

## 🌐 Application Access

Once running, access these interfaces:

- **📊 Main Dashboard**: http://localhost:5000
- **🤖 ML Training**: http://localhost:5000 → "ML Training" tab
- **📈 Backtesting**: http://localhost:5000 → "Backtesting" tab
- **⚙️ Configuration**: http://localhost:5000 → "Configuration" tab
- **🚀 REST API**: http://localhost:8000 (run `python api/server.py`)

## 🛠️ Advanced Usage

### Start API Server
```cmd
python api/server.py
```

### Manual Model Training
1. Go to "ML Training" tab in the dashboard
2. Click "🔄 Trigger Retraining"
3. Monitor performance metrics in real-time

### Run Backtesting
1. Go to "Backtesting" tab in the dashboard  
2. Select date range and minimum impact score
3. Click "🚀 Run Backtest" for historical analysis

### Data Export
- Catalysts and analysis results are automatically saved to SQLite database
- Export functionality available through the dashboard interface
- API endpoints provide programmatic data access

## 🔧 Troubleshooting

### Installation Issues
- Ensure Python 3.9+ is installed with PATH configured
- Run `python --version` to verify installation
- Try running `pip install --upgrade pip` if package installation fails

### Database Issues
- Uses SQLite by default (no additional setup required)
- Database file created automatically in `data/financial_scanner.db`
- For PostgreSQL: Update DATABASE_URL in `.env` file

### Network Issues
- Check firewall settings for port 5000
- Ensure internet connectivity for data source APIs
- Verify API keys are correctly configured

## 📞 Support

For technical support or feature requests:
1. Check the Configuration tab for API key setup
2. Review logs in the System Status tab
3. Ensure all required dependencies are installed

---

**🎉 You now have a complete, enterprise-grade financial intelligence platform running locally on Windows 11!**