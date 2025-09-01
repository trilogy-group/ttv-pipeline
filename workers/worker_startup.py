#!/usr/bin/env python3
"""
Worker startup script that initializes the queue infrastructure
before starting the RQ worker.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config import get_config_from_env
from api.queue import initialize_queue_infrastructure
from api.logging_config import setup_logging

def initialize_worker():
    """Initialize worker with proper queue infrastructure."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing worker...")
    
    # Load configuration
    config = get_config_from_env()
    logger.info(f"Loaded configuration for environment: {config.environment}")
    
    # Initialize queue infrastructure
    redis_manager, job_queue = initialize_queue_infrastructure(config.redis)
    logger.info("Queue infrastructure initialized successfully")
    
    logger.info("Worker initialization complete")
    return True

if __name__ == "__main__":
    initialize_worker()