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

## Installation
```
pip install -r requirements.txt
```

## Usage
### Running the API
```
cd MemoTag
uvicorn app.main:app --reload
```

### Accessing the Endpoint
The API will be available at `http://localhost:8000`

### API Documentation
Swagger UI: `http://localhost:8000/docs`

## Report
For detailed analysis and findings, see the jupyter notebook in the `notebooks` directory.
