"""
Mock GCS client and uploader for integration testing.
"""

import os
import time
from typing import Dict, Any, Optional
from unittest.mock import Mock

from api.exceptions import GCSError


class MockGCSClient:
    """Mock GCS client that simulates Google Cloud Storage operations"""
    
    def __init__(self, config):
        self.config = config
        self.uploaded_files = {}  # path -> content mapping
        self.upload_count = 0
        self.download_count = 0
        self.signed_url_count = 0
        
    def upload_artifact(self, local_path: str, job_id: str) -> str:
        """Mock artifact upload"""
        if not os.path.exists(local_path):
            raise GCSError(f"Local file not found: {local_path}")
        
        # Generate GCS path
        gcs_path = self.generate_artifact_path(job_id)
        
        # Read and store file content
        with open(local_path, 'rb') as f:
            content = f.read()
        
        self.uploaded_files[gcs_path] = {
            'content': content,
            'size': len(content),
            'uploaded_at': time.time(),
            'job_id': job_id
        }
        
        self.upload_count += 1
        return gcs_path
    
    def generate_artifact_path(self, job_id: str) -> str:
        """Generate GCS path following the required format"""
        from datetime import datetime
        
        # Format: gs://{bucket}/{prefix}/{YYYY-MM}/{job_id}/final_video.mp4
        now = datetime.utcnow()
        year_month = now.strftime("%Y-%m")
        
        return f"gs://{self.config.bucket}/{self.config.prefix}/{year_month}/{job_id}/final_video.mp4"
    
    def generate_signed_url(self, gcs_uri: str, expiration_seconds: int = 3600) -> str:
        """Mock signed URL generation"""
        # Extract path from GCS URI
        if not gcs_uri.startswith("gs://"):
            raise GCSError(f"Invalid GCS URI: {gcs_uri}")
        
        path = gcs_uri.replace(f"gs://{self.config.bucket}/", "")
        
        if gcs_uri not in self.uploaded_files:
            raise GCSError(f"File not found in GCS: {gcs_uri}")
        
        self.signed_url_count += 1
        
        # Generate mock signed URL
        expires_at = int(time.time()) + expiration_seconds
        return f"https://storage.googleapis.com/{self.config.bucket}/{path}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Expires={expiration_seconds}&X-Goog-SignedHeaders=host&expires={expires_at}"
    
    def download_artifact(self, gcs_uri: str, local_path: str) -> str:
        """Mock artifact download"""
        if gcs_uri not in self.uploaded_files:
            raise GCSError(f"File not found in GCS: {gcs_uri}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Write file content
        file_info = self.uploaded_files[gcs_uri]
        with open(local_path, 'wb') as f:
            f.write(file_info['content'])
        
        self.download_count += 1
        return local_path
    
    def file_exists(self, gcs_uri: str) -> bool:
        """Check if file exists in mock GCS"""
        return gcs_uri in self.uploaded_files
    
    def get_file_info(self, gcs_uri: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        return self.uploaded_files.get(gcs_uri)
    
    def delete_file(self, gcs_uri: str) -> bool:
        """Delete file from mock GCS"""
        if gcs_uri in self.uploaded_files:
            del self.uploaded_files[gcs_uri]
            return True
        return False
    
    def list_files(self, prefix: str = "") -> list:
        """List files with given prefix"""
        prefix_filter = f"gs://{self.config.bucket}/{prefix}" if prefix else f"gs://{self.config.bucket}/"
        return [uri for uri in self.uploaded_files.keys() if uri.startswith(prefix_filter)]
    
    def get_stats(self) -> Dict[str, int]:
        """Get operation statistics"""
        return {
            "upload_count": self.upload_count,
            "download_count": self.download_count,
            "signed_url_count": self.signed_url_count,
            "total_files": len(self.uploaded_files)
        }
    
    def reset(self):
        """Reset mock state"""
        self.uploaded_files.clear()
        self.upload_count = 0
        self.download_count = 0
        self.signed_url_count = 0


class MockGCSUploader:
    """Mock GCS uploader for worker processes"""
    
    def __init__(self, gcs_client: MockGCSClient):
        self.gcs_client = gcs_client
        self.upload_progress_callback = None
    
    def upload_with_progress(self, local_path: str, job_id: str, 
                           progress_callback=None) -> str:
        """Upload with progress reporting"""
        self.upload_progress_callback = progress_callback
        
        if progress_callback:
            progress_callback(0, "Starting upload...")
            time.sleep(0.1)
            progress_callback(25, "Uploading...")
            time.sleep(0.1)
            progress_callback(75, "Finalizing...")
            time.sleep(0.1)
            progress_callback(100, "Upload complete")
        
        return self.gcs_client.upload_artifact(local_path, job_id)
    
    def set_progress_callback(self, callback):
        """Set progress callback"""
        self.upload_progress_callback = callback


class MockGCSError(Exception):
    """Mock GCS error for testing error handling"""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN"):
        super().__init__(message)
        self.error_code = error_code


def create_failing_gcs_client(config, failure_type: str = "upload"):
    """Create a GCS client that fails in specific ways"""
    
    class FailingMockGCSClient(MockGCSClient):
        def __init__(self, config, failure_type):
            super().__init__(config)
            self.failure_type = failure_type
        
        def upload_artifact(self, local_path: str, job_id: str) -> str:
            if self.failure_type == "upload":
                raise GCSError("Mock upload failure")
            return super().upload_artifact(local_path, job_id)
        
        def generate_signed_url(self, gcs_uri: str, expiration: int = 3600) -> str:
            if self.failure_type == "signed_url":
                raise GCSError("Mock signed URL generation failure")
            return super().generate_signed_url(gcs_uri, expiration)
        
        def download_artifact(self, gcs_uri: str, local_path: str) -> str:
            if self.failure_type == "download":
                raise GCSError("Mock download failure")
            return super().download_artifact(gcs_uri, local_path)
    
    return FailingMockGCSClient(config, failure_type)


def create_slow_gcs_client(config, delay_seconds: float = 2.0):
    """Create a GCS client with artificial delays"""
    
    class SlowMockGCSClient(MockGCSClient):
        def __init__(self, config, delay):
            super().__init__(config)
            self.delay = delay
        
        def upload_artifact(self, local_path: str, job_id: str) -> str:
            time.sleep(self.delay)
            return super().upload_artifact(local_path, job_id)
        
        def generate_signed_url(self, gcs_uri: str, expiration: int = 3600) -> str:
            time.sleep(self.delay / 2)  # Shorter delay for URL generation
            return super().generate_signed_url(gcs_uri, expiration)
    
    return SlowMockGCSClient(config, delay_seconds)