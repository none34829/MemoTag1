"""
Script to run the MemoTag cognitive decline detection API.
"""

import os
import sys
import logging
import argparse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_data_available():
    """Ensure sample data is available for the application."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    
    # Check if data directory exists and has files
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.info("No sample data found. Downloading test samples...")
        
        # Import and run the data downloader
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from src.data_downloader import get_small_test_files
            
            # Download small test files
            get_small_test_files(data_dir)
            logger.info("Sample data downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading sample data: {e}")
            logger.warning("API will start, but you may need to provide your own audio files for testing")
    else:
        logger.info(f"Found existing data in {data_dir}")

def run_api(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI application."""
    logger.info(f"Starting MemoTag Cognitive Decline Detection API on http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Ensure we have some sample data
    ensure_data_available()
    
    # Run the API
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MemoTag Cognitive Decline Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload on code changes")
    
    args = parser.parse_args()
    run_api(host=args.host, port=args.port, reload=not args.no_reload)
