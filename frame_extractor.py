"""
Frame extraction utilities for chaining mode in the TTV pipeline.
"""

import os
import logging
import subprocess
from typing import Optional

def extract_last_frame(video_path: str, output_path: str) -> Optional[str]:
    """
    Extract the last frame from a video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_path: Path where the extracted frame should be saved
        
    Returns:
        Path to the extracted frame if successful, None otherwise
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file does not exist: {video_path}")
        return None
        
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Use ffmpeg to extract the last frame
        # -sseof -0.1: Seek to 0.1 seconds before the end of the file
        # -vframes 1: Extract only one frame
        # -q:v 2: Set quality level (lower is better, 2-5 is good range for PNG)
        cmd = [
            "ffmpeg", 
            "-sseof", "-0.1", 
            "-i", video_path, 
            "-vframes", "1", 
            "-q:v", "2", 
            output_path
        ]
        
        logging.info(f"Extracting last frame from {video_path} to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Frame extraction failed: {result.stderr}")
            return None
            
        if os.path.exists(output_path):
            logging.info(f"Successfully extracted frame to {output_path}")
            return output_path
        else:
            logging.error("Frame extraction completed but output file doesn't exist")
            return None
            
    except Exception as e:
        logging.error(f"Error during frame extraction: {e}")
        return None
