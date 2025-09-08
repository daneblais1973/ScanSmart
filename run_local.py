#!/usr/bin/env python3
"""
Local launcher for Financial Catalyst Scanner on Windows 11
Handles all setup and configuration automatically
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        import importlib
        importlib.import_module(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def get_installed_version(package_name):
    """Get the installed version of a package"""
    try:
        # Use importlib.metadata instead of deprecated pkg_resources
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            return None
    except:
        return None

def parse_requirement(requirement):
    """Parse package requirement string"""
    import re
    match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+)?(.+)?$', requirement)
    if match:
        name = match.group(1)
        operator = match.group(2) or '>='
        version = match.group(3) or '0.0.0'
        return name, operator, version
    return requirement, '>=', '0.0.0'

def check_version_compatibility(installed_version, required_operator, required_version):
    """Check if installed version meets requirements"""
    if not installed_version:
        return False
    
    try:
        from packaging import version
        installed = version.parse(installed_version)
        required = version.parse(required_version)
        
        if required_operator == '>=':
            return installed >= required
        elif required_operator == '>':
            return installed > required
        elif required_operator == '==':
            return installed == required
        elif required_operator == '<=':
            return installed <= required
        elif required_operator == '<':
            return installed < required
        elif required_operator == '!=':
            return installed != required
    except:
        return True  # If version parsing fails, assume compatible
    
    return True

def install_dependencies():
    """Install required packages with smart dependency checking"""
    print("📦 Checking and installing dependencies...")
    
    required_packages = [
        "streamlit>=1.28.0",
        "fastapi>=0.100.0", 
        "uvicorn>=0.22.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "aiohttp>=3.8.0",
        "requests>=2.31.0",
        "trafilatura>=1.6.0",
        "feedparser>=6.0.0",
        "plotly>=5.15.0",
        "pyjwt>=2.8.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "schedule>=1.2.0",
        "tenacity>=8.2.0",
        "python-dateutil>=2.8.0",
        "twilio>=8.5.0",
        "yfinance>=0.2.0",
        "textblob>=0.17.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.5.0",
        "openai>=1.0.0",
        "psutil>=5.9.0"
    ]
    
    # First, install packaging for version checking
    try:
        import packaging
    except ImportError:
        print("Installing packaging for version checking...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    
    packages_to_install = []
    packages_to_upgrade = []
    conflicts_found = []
    
    print("🔍 Analyzing existing packages...")
    
    for requirement in required_packages:
        package_name, operator, version = parse_requirement(requirement)
        installed_version = get_installed_version(package_name)
        
        if installed_version:
            is_compatible = check_version_compatibility(installed_version, operator, version)
            if is_compatible:
                print(f"✅ {package_name} {installed_version} (compatible)")
            else:
                print(f"⚠️  {package_name} {installed_version} -> needs {operator}{version}")
                packages_to_upgrade.append(requirement)
                conflicts_found.append(f"{package_name}: {installed_version} -> {operator}{version}")
        else:
            print(f"📦 {package_name} (not installed)")
            packages_to_install.append(requirement)
    
    # Report findings
    if conflicts_found:
        print(f"\n🔄 Found {len(conflicts_found)} packages that need upgrading:")
        for conflict in conflicts_found:
            print(f"   - {conflict}")
    
    if packages_to_install:
        print(f"\n📦 Found {len(packages_to_install)} new packages to install")
    
    if not packages_to_install and not packages_to_upgrade:
        print("✅ All dependencies are already satisfied!")
        return True
    
    # Install new packages
    if packages_to_install:
        print(f"\n📦 Installing {len(packages_to_install)} new packages...")
        try:
            for package in packages_to_install:
                print(f"   Installing {package}...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                      capture_output=True, text=True, check=True)
            print("✅ New packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install new packages: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error details: {e.stderr}")
            return False
    
    # Upgrade existing packages
    if packages_to_upgrade:
        print(f"\n🔄 Upgrading {len(packages_to_upgrade)} existing packages...")
        try:
            for package in packages_to_upgrade:
                print(f"   Upgrading {package}...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], 
                                      capture_output=True, text=True, check=True)
            print("✅ Package upgrades completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade packages: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error details: {e.stderr}")
            print("⚠️  Some upgrades failed, but installation will continue...")
    
    # Final verification
    print("\n🔍 Verifying installations...")
    verification_failed = []
    
    for requirement in required_packages:
        package_name, operator, version = parse_requirement(requirement)
        installed_version = get_installed_version(package_name)
        
        if not installed_version:
            verification_failed.append(package_name)
        elif not check_version_compatibility(installed_version, operator, version):
            verification_failed.append(f"{package_name} (version conflict)")
    
    if verification_failed:
        print(f"⚠️  Verification issues found:")
        for issue in verification_failed:
            print(f"   - {issue}")
        print("🔧 Some packages may not meet exact version requirements, but the app should still work.")
    else:
        print("✅ All dependencies verified successfully!")
    
    return True

def setup_environment():
    """Setup environment variables and directories"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "data/cache",
        "data/logs", 
        "data/exports",
        "models/trained",
        "backtesting/results",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create default .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Financial Catalyst Scanner Configuration
# Database Configuration (SQLite for local development)
DATABASE_URL=sqlite:///./data/financial_scanner.db

# Redis Configuration (Optional - will use memory cache if not available)
REDIS_URL=redis://localhost:6379

# API Keys (Configure through the UI interface)
NEWSAPI_KEY=
TWITTER_BEARER_TOKEN=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
OPENAI_API_KEY=

# Twilio Configuration (For SMS alerts)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

# Application Settings
LOG_LEVEL=INFO
CACHE_TIMEOUT=3600
MAX_WORKERS=4
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Environment configuration created!")
    else:
        print("✅ Environment configuration already exists!")
    
    return True

def create_streamlit_config():
    """Create Streamlit configuration for local deployment"""
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.toml"
    config_content = """[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "light"
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    print("✅ Streamlit configuration created!")
    
    return True

def start_application():
    """Start the Financial Catalyst Scanner application"""
    print("🚀 Starting Financial Catalyst Scanner...")
    
    try:
        # Start Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "5000"]
        
        print("📊 Application starting on http://localhost:5000")
        print("🌐 Opening browser automatically in 5 seconds...")
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a moment then open browser
        time.sleep(5)
        webbrowser.open("http://localhost:5000")
        
        print("\n" + "="*60)
        print("🎉 FINANCIAL CATALYST SCANNER RUNNING")
        print("="*60)
        print("📊 Dashboard: http://localhost:5000")
        print("🔧 Configuration: Go to 'Configuration' tab to set API keys")
        print("🤖 ML Training: Use 'ML Training' tab for model management")
        print("📈 Backtesting: Use 'Backtesting' tab for historical analysis")
        print("🚀 API Server: Run 'python api/server.py' for REST API")
        print("\n💡 Press Ctrl+C to stop the application")
        print("="*60)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping Financial Catalyst Scanner...")
        process.terminate()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("="*60)
    print("🎯 FINANCIAL CATALYST SCANNER - LOCAL SETUP")
    print("="*60)
    print("🔥 Advanced Financial Intelligence Platform")
    print("✨ Features: AI Analysis, Real-time Monitoring, ML Training")
    print("📊 Components: Dashboard, API, Backtesting, Alerts")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment), 
        ("Creating Streamlit config", create_streamlit_config),
        ("Starting application", start_application)
    ]
    
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}...")
        if not step_function():
            print(f"❌ Failed at step: {step_name}")
            sys.exit(1)

if __name__ == "__main__":
    main()