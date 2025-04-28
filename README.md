# MemoTag Voice-Based Cognitive Decline Detection

## Overview
This project implements a proof-of-concept pipeline for detecting cognitive decline indicators from voice data. It extracts audio features and linguistic patterns that might signal early cognitive impairment.

## Features
- Audio preprocessing and feature extraction
- Text transcription and linguistic analysis
- Detection of cognitive decline markers:
  - Speech hesitations and pauses
  - Word recall issues
  - Speech rate and rhythm changes
  - Naming difficulties
  - Sentence completion challenges
- Unsupervised ML approach for pattern detection
- API endpoint for real-time scoring

## Project Structure
```
MemoTag/
├── app/                     # API application
│   ├── main.py              # FastAPI application
│   ├── models/              # ML models
│   └── utils/               # Utility functions
├── data/                    # Data storage
│   ├── raw/                 # Raw audio files
│   └── processed/           # Processed features
├── notebooks/               # Analysis notebooks
├── src/                     # Source code
│   ├── audio_processing.py  # Audio feature extraction
│   ├── text_processing.py   # Text feature extraction
│   ├── model.py             # ML model implementation
│   └── visualization.py     # Visualization utilities
├── tests/                   # Test files
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Detailed Setup Guide for Local Development

### Prerequisites
- Python 3.9+ installed
- Git installed
- Sufficient disk space (approximately 500MB for dependencies)
- Audio playback capabilities (for testing)

### Step 1: Clone the Repository
```bash
# Clone the repository from GitHub
git clone https://github.com/none34829/MemoTag.git

# Navigate to the project directory
cd MemoTag
```

### Step 2: Set Up Virtual Environment
It's highly recommended to use a virtual environment to avoid dependency conflicts.

#### For Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### For macOS/Linux:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements-no-hash.txt
```

This will install all necessary packages including:
- Core ML packages (numpy, pandas, scikit-learn)
- Audio processing libraries (librosa, pydub)
- Speech recognition tools (SpeechRecognition, whisper)
- NLP libraries (nltk, spacy, transformers)
- API framework (FastAPI, uvicorn)

### Step 4: Download Additional Resources (if needed)
Some packages may require additional resources:

```bash
# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 5: Create Required Directories
```bash
# Create necessary directories if they don't exist
mkdir -p data/raw data/processed models
```

### Step 6: Run the Application
```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be accessible at `http://localhost:8000`

### Step 7: Verify Installation
- Open your web browser and navigate to `http://localhost:8000`
- You should see a JSON response with API information
- Visit `http://localhost:8000/docs` to access the Swagger UI documentation

### Testing the API
You can test the API endpoint by uploading an audio file:

1. Go to `http://localhost:8000/docs`
2. Expand the `/analyze` endpoint
3. Click "Try it out"
4. Upload an audio file (WAV, MP3, OGG, FLAC, or M4A format)
5. Click "Execute"
6. View the response with the cognitive decline risk assessment

### Troubleshooting Common Issues

#### Missing Dependencies
If you encounter errors about missing modules:
```bash
pip install -r requirements-no-hash.txt --no-cache-dir
```

#### CUDA/GPU Issues
For GPU acceleration (optional):
```bash
# Install PyTorch with CUDA support (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Audio Processing Issues
If you encounter audio processing errors:
- Ensure FFmpeg is installed on your system
- For Windows: Install from https://ffmpeg.org/download.html and add to PATH
- For macOS: `brew install ffmpeg`
- For Linux: `sudo apt-get install ffmpeg`

#### Port Already in Use
If port 8000 is already in use:
```bash
# Use a different port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

### Running in Production Mode
For production deployment:
```bash
# Run without the --reload flag
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation
Swagger UI: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

## Report
For detailed analysis and findings, see the jupyter notebook in the `notebooks` directory.
