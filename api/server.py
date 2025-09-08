#!/usr/bin/env python3
"""
FastAPI server launcher for Financial Catalyst Scanner API
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AppConfig
from api.endpoints import CatalystScannerAPI

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main server entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config = AppConfig()
        
        # Create API instance
        api = CatalystScannerAPI(config)
        
        # Run server
        logger.info("Starting Financial Catalyst Scanner API server...")
        api.run(host="0.0.0.0", port=8000)
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()