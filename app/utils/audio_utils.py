"""
Utility functions for handling audio files in the API.
"""

import os
import tempfile
import uuid
from typing import List, Dict, Any, BinaryIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_uploaded_file(file: BinaryIO, upload_dir: str = None) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        file: File-like object containing audio data
        upload_dir: Directory to save the file, uses temp dir if None
        
    Returns:
        Path to saved file
    """
    # Create upload directory if it doesn't exist
    if upload_dir is None:
        upload_dir = tempfile.gettempdir()
    elif not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(upload_dir, filename)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(file.read())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def clean_up_file(file_path: str) -> bool:
    """
    Remove a temporary file after processing.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {e}")
        return False
