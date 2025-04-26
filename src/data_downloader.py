"""
Data downloader script to get sample audio files for cognitive decline analysis.
"""

import os
import requests
import zipfile
import io
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define URLs for sample datasets
# These are public domain or CC-licensed audio samples that can be used for testing
SAMPLE_DATA_URLS = {
    "commonvoice": "https://storage.googleapis.com/common-voice-data-download/cv-corpus-5-singleword/cv-corpus-5-singleword-en.tar.gz",
    "librispeech_samples": "https://www.openslr.org/resources/12/dev-clean-2.tar.gz",
    "dementiabank_samples": "https://media.talkbank.org/dementia/English/Pitt/dementia_files.zip",
}

# Alternative: Use these URLs for smaller test files
SMALL_TEST_FILES = {
    "sample1": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav",
    "sample2": "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav",
    "sample3": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    "sample4": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
    "sample5": "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav",
}

def download_file(url, destination):
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        
    Returns:
        Path to downloaded file
    """
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Write file with progress bar
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        logger.info(f"Downloaded to {destination}")
        return destination
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise

def extract_archive(archive_path, extract_dir):
    """
    Extract an archive file.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        
    Returns:
        Path to extraction directory
    """
    try:
        logger.info(f"Extracting {archive_path}...")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return None
        
        logger.info(f"Extracted to {extract_dir}")
        return extract_dir
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        raise

def get_small_test_files(data_dir):
    """
    Download small test WAV files for quick testing.
    
    Args:
        data_dir: Directory to save files
        
    Returns:
        List of downloaded file paths
    """
    os.makedirs(data_dir, exist_ok=True)
    downloaded_files = []
    
    for name, url in SMALL_TEST_FILES.items():
        destination = os.path.join(data_dir, f"{name}.wav")
        
        try:
            download_file(url, destination)
            downloaded_files.append(destination)
        except Exception as e:
            logger.error(f"Error downloading test file {name}: {e}")
    
    return downloaded_files

def download_dataset(dataset_name, data_dir):
    """
    Download and extract a dataset.
    
    Args:
        dataset_name: Name of the dataset to download
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the extracted dataset
    """
    if dataset_name not in SAMPLE_DATA_URLS:
        logger.error(f"Dataset {dataset_name} not found")
        return None
    
    url = SAMPLE_DATA_URLS[dataset_name]
    os.makedirs(data_dir, exist_ok=True)
    
    # Download archive
    archive_name = os.path.basename(url)
    archive_path = os.path.join(data_dir, archive_name)
    
    try:
        # Download the file
        download_file(url, archive_path)
        
        # Extract the archive
        extract_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(extract_dir, exist_ok=True)
        extract_archive(archive_path, extract_dir)
        
        # Remove the archive file to save space
        os.remove(archive_path)
        
        return extract_dir
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {e}")
        return None

def main():
    """Main function to download sample data."""
    # Define data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # For quick testing, just download small test files
    logger.info("Downloading small test files...")
    test_files = get_small_test_files(data_dir)
    logger.info(f"Downloaded {len(test_files)} test files")
    
    # Uncomment to download larger datasets
    # logger.info("Downloading LibriSpeech samples...")
    # librispeech_dir = download_dataset("librispeech_samples", data_dir)
    
    # logger.info("Downloading DementiaBank samples...")
    # dementiabank_dir = download_dataset("dementiabank_samples", data_dir)

if __name__ == "__main__":
    main()
