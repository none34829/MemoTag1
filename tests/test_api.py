"""
Test script for the MemoTag cognitive decline detection API.
"""

import os
import sys
import requests
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_health(base_url="http://localhost:8000"):
    """Test the API health endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            logger.info("Health check passed")
            return True
        else:
            logger.error(f"Health check failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")
        return False

def test_audio_analysis(audio_file_path, base_url="http://localhost:8000"):
    """Test the audio analysis endpoint."""
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return False
    
    try:
        # Open the file in binary mode
        with open(audio_file_path, 'rb') as file:
            # Create a multipart form with the file
            files = {
                'file': (os.path.basename(audio_file_path), file, 'audio/wav')
            }
            
            # Send the request
            logger.info(f"Sending {audio_file_path} to API for analysis...")
            response = requests.post(f"{base_url}/analyze", files=files)
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis successful: Risk score {result['risk_score']}, Level: {result['risk_level']}")
                logger.info(f"Transcript: {result['transcript'][:100]}...")
                
                # Print key indicators
                logger.info("Key indicators:")
                for indicator in result['key_indicators']:
                    logger.info(f"  - {indicator['feature']}: {indicator['importance']:.3f}")
                
                return result
            else:
                logger.error(f"Analysis failed: {response.status_code} {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        return False

def test_batch_training(audio_file_paths, base_url="http://localhost:8000"):
    """Test the batch training endpoint."""
    if not audio_file_paths:
        logger.error("No audio files provided for batch training")
        return False
    
    # Check that all files exist
    for path in audio_file_paths:
        if not os.path.exists(path):
            logger.error(f"Audio file not found: {path}")
            return False
    
    try:
        # Create a multipart form with multiple files
        files = []
        for path in audio_file_paths:
            with open(path, 'rb') as file:
                files.append(('files', (os.path.basename(path), file, 'audio/wav')))
        
        # Send the request
        logger.info(f"Sending {len(files)} files to API for batch training...")
        response = requests.post(f"{base_url}/batch-train", files=files)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            logger.info("Batch training successful")
            return result
        else:
            logger.error(f"Batch training failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error in batch training: {e}")
        return False

def run_all_tests():
    """Run all API tests."""
    # Define base URL
    base_url = "http://localhost:8000"
    
    # Check if the API is running
    if not test_api_health(base_url):
        logger.error("API health check failed. Make sure the API is running.")
        return False
    
    # Get audio files for testing
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]
    
    if not audio_files:
        logger.error(f"No audio files found in {data_dir}. Please download sample files first.")
        return False
    
    # Run batch training with all files
    logger.info("Testing batch training...")
    batch_result = test_batch_training(audio_files, base_url)
    
    # Test individual file analysis
    logger.info("Testing individual file analysis...")
    for audio_file in audio_files:
        test_audio_analysis(audio_file, base_url)
    
    logger.info("All tests completed")
    return True

if __name__ == "__main__":
    run_all_tests()
