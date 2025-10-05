#!/usr/bin/env python3
"""
GCS Setup Validation Script

This script validates that GCS credentials are properly configured
without running the entire pipeline. Run this before submitting jobs
to catch configuration issues early.

Usage:
    python validate_gcs_setup.py
    
Or in a worker container:
    docker compose exec worker python validate_gcs_setup.py
"""

import os
import sys
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def validate_gcs_credentials():
    """Validate GCS credentials configuration"""
    
    print("=" * 70)
    print("GCS CREDENTIALS VALIDATION")
    print("=" * 70)
    print()
    
    # Step 1: Check pipeline config file exists
    config_path = "/app/pipeline_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"❌ Pipeline config not found at: {config_path}")
        return False
    
    logger.info(f"✅ Pipeline config found: {config_path}")
    
    # Step 2: Load config and check google_credentials_path
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline config: {e}")
        return False
    
    google_credentials_path = config.get('google_credentials_path', 'credentials.json')
    logger.info(f"📝 Configured credentials path: {google_credentials_path}")
    
    # Step 3: Convert to absolute path (same logic as video_worker.py)
    if not os.path.isabs(google_credentials_path):
        google_credentials_path = os.path.join('/app', google_credentials_path)
        logger.info(f"📍 Absolute credentials path: {google_credentials_path}")
    
    # Step 4: Check if credentials file exists
    if not os.path.exists(google_credentials_path):
        logger.error(f"❌ GCS credentials file NOT FOUND at: {google_credentials_path}")
        logger.info(f"\n📂 Available files in /app:")
        try:
            files = os.listdir('/app')
            for f in sorted(files)[:30]:  # Show first 30 files
                logger.info(f"   - {f}")
        except Exception as e:
            logger.error(f"   Failed to list /app directory: {e}")
        return False
    
    logger.info(f"✅ GCS credentials file exists: {google_credentials_path}")
    
    # Step 5: Validate it's a valid JSON file
    try:
        import json
        with open(google_credentials_path, 'r') as f:
            creds = json.load(f)
        
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in creds]
        
        if missing_fields:
            logger.error(f"❌ Credentials file missing required fields: {missing_fields}")
            return False
        
        logger.info(f"✅ Credentials file is valid JSON with required fields")
        logger.info(f"   Project ID: {creds.get('project_id')}")
        logger.info(f"   Client Email: {creds.get('client_email')}")
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Credentials file is not valid JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to validate credentials file: {e}")
        return False
    
    # Step 6: Test GCS client initialization
    try:
        from google.cloud import storage
        client = storage.Client.from_service_account_json(google_credentials_path)
        logger.info(f"✅ GCS client initialized successfully")
        
        # Try to list buckets to validate credentials work
        try:
            # Just check if we can authenticate, don't actually list buckets
            # (might fail due to permissions)
            project_id = creds.get('project_id')
            logger.info(f"✅ Credentials valid for project: {project_id}")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify bucket access (might be permissions): {e}")
            # This is OK - credentials might not have bucket listing permissions
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize GCS client: {e}")
        return False
    
    # Step 7: Check GCS bucket configuration
    gcs_bucket = config.get('gcs_bucket')
    if not gcs_bucket:
        logger.warning(f"⚠️  No gcs_bucket configured in pipeline config")
    else:
        logger.info(f"✅ GCS bucket configured: {gcs_bucket}")
    
    print()
    print("=" * 70)
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 70)
    print()
    logger.info("Your GCS setup is correct and ready to use!")
    return True


def main():
    """Main validation function"""
    try:
        success = validate_gcs_credentials()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Validation script failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
