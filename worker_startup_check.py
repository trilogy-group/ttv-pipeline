#!/usr/bin/env python3
"""
Worker Startup Validation - MANDATORY CHECK

This script runs on worker startup to validate critical configuration
before accepting any jobs. If validation fails, the worker will NOT start.

This prevents wasting 30+ minutes on jobs that will fail at the final step.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def fatal_error(message: str):
    """Log fatal error and exit"""
    logger.error("=" * 80)
    logger.error("FATAL: WORKER STARTUP VALIDATION FAILED")
    logger.error("=" * 80)
    logger.error(message)
    logger.error("=" * 80)
    logger.error("Worker will NOT start until this is fixed!")
    logger.error("=" * 80)
    sys.exit(1)


def validate_gcs_credentials():
    """Validate GCS credentials - MANDATORY for worker startup"""
    
    logger.info("=" * 80)
    logger.info("WORKER STARTUP VALIDATION - GCS CREDENTIALS CHECK")
    logger.info("=" * 80)
    
    # Step 1: Check pipeline config exists
    config_path = "/app/pipeline_config.yaml"
    if not os.path.exists(config_path):
        fatal_error(f"Pipeline config not found at: {config_path}")
    
    logger.info(f"✅ Pipeline config found: {config_path}")
    
    # Step 2: Load and parse config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        fatal_error(f"Failed to load pipeline config: {e}")
    
    # Step 3: Get google_credentials_path
    google_credentials_path = config.get('google_credentials_path')
    if not google_credentials_path:
        fatal_error(
            "google_credentials_path not configured in pipeline_config.yaml!\n"
            "Add: google_credentials_path: \"your-credentials-file.json\""
        )
    
    logger.info(f"📝 Configured credentials path: {google_credentials_path}")
    
    # Step 4: Convert to absolute path (same logic as video_worker.py)
    if not os.path.isabs(google_credentials_path):
        google_credentials_path = os.path.join('/app', google_credentials_path)
    
    logger.info(f"📍 Absolute credentials path: {google_credentials_path}")
    
    # Step 5: CRITICAL - Verify file exists
    if not os.path.exists(google_credentials_path):
        available_files = os.listdir('/app')[:30]
        fatal_error(
            f"GCS credentials file NOT FOUND: {google_credentials_path}\n\n"
            f"Available files in /app:\n" +
            "\n".join(f"  - {f}" for f in sorted(available_files)) +
            "\n\nFIX: Ensure the credentials file is mounted/copied to the container!"
        )
    
    logger.info(f"✅ GCS credentials file exists")
    
    # Step 6: Validate JSON structure
    try:
        import json
        with open(google_credentials_path, 'r') as f:
            creds = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds]
        
        if missing_fields:
            fatal_error(
                f"Credentials file is missing required fields: {missing_fields}\n"
                f"File: {google_credentials_path}"
            )
        
        logger.info(f"✅ Credentials file is valid JSON")
        logger.info(f"   Project: {creds.get('project_id')}")
        logger.info(f"   Email: {creds.get('client_email')}")
        
    except json.JSONDecodeError as e:
        fatal_error(f"Credentials file is not valid JSON: {e}")
    except Exception as e:
        fatal_error(f"Failed to validate credentials file: {e}")
    
    # Step 7: Test GCS client initialization
    try:
        from google.cloud import storage
        client = storage.Client.from_service_account_json(google_credentials_path)
        logger.info(f"✅ GCS client initialized successfully")
    except Exception as e:
        fatal_error(f"Failed to initialize GCS client: {e}")
    
    # Step 8: Check bucket configuration
    gcs_bucket = config.get('gcs_bucket')
    if not gcs_bucket:
        fatal_error(
            "gcs_bucket not configured in pipeline_config.yaml!\n"
            "Add: gcs_bucket: \"your-bucket-name\""
        )
    
    logger.info(f"✅ GCS bucket configured: {gcs_bucket}")
    
    logger.info("=" * 80)
    logger.info("✅ ALL STARTUP VALIDATION CHECKS PASSED")
    logger.info("=" * 80)
    logger.info("Worker is ready to process jobs!")
    logger.info("=" * 80)


def main():
    """Main validation entry point"""
    try:
        validate_gcs_credentials()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        fatal_error(f"Unexpected validation error: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
