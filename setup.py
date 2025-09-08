from setuptools import setup, find_packages

setup(
    name="financial-catalyst-scanner",
    version="2.0.0",
    description="Advanced Financial Catalyst Scanning and Intelligence Platform",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    author="Financial Intelligence Systems",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        # Core web framework
        "streamlit>=1.28.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        
        # Database and caching
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.5.0",
        "alembic>=1.11.0",
        
        # Data processing
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        
        # ML and NLP (optional - fallbacks provided)
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "spacy>=3.6.0",
        
        # API and networking
        "aiohttp>=3.8.0",
        "httpx>=0.24.0",
        "requests>=2.31.0",
        
        # Web scraping
        "trafilatura>=1.6.0",
        "feedparser>=6.0.0",
        
        # Visualization
        "plotly>=5.15.0",
        
        # Authentication and security
        "pyjwt>=2.8.0",
        "python-multipart>=0.0.6",
        
        # Configuration and environment
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        
        # Scheduling and background tasks
        "schedule>=1.2.0",
        
        # Utilities
        "tenacity>=8.2.0",
        "python-dateutil>=2.8.0",
        
        # Communication and alerting
        "twilio>=8.5.0",
    ],
    extras_require={
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "catalyst-scanner=app:main",
            "catalyst-api=api.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)