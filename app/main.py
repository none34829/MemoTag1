"""
FastAPI application for the MemoTag cognitive decline detection API.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Import scipy compatibility layer to handle version differences
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scipy_compat import *  # This adds hann function to scipy.signal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.audio_utils import save_uploaded_file, clean_up_file
from src.audio_processing import AudioProcessor
from src.text_processing import TextProcessor
from src.model import CognitiveDeclineModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define app
app = FastAPI(
    title="MemoTag Cognitive Decline Detection API",
    description="API for detecting cognitive decline indicators from voice data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize processors and model
audio_processor = AudioProcessor()
text_processor = TextProcessor()

# Set up model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model = CognitiveDeclineModel(model_dir=MODEL_DIR)

# Try to load pre-trained model if available
try:
    model_loaded = model.load_models()
    if model_loaded:
        logger.info("Pre-trained model loaded successfully")
    else:
        logger.warning("No pre-trained model found. Will train on first batch of data.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.warning("Will train model on first batch of data")

# Response model for risk score
class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_level: str
    transcript: str
    key_indicators: List[Dict[str, Any]]
    audio_features: Dict[str, float]
    text_features: Dict[str, float]

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MemoTag Cognitive Decline Detection API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/analyze", response_model=RiskScoreResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for cognitive decline indicators.
    
    - **file**: Audio file (WAV format recommended)
    
    Returns a risk score and analysis results.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
        raise HTTPException(status_code=400, 
                           detail="Unsupported file format. Please upload WAV, MP3, OGG, FLAC, or M4A.")
    
    try:
        # Save uploaded file
        temp_file_path = save_uploaded_file(file.file)
        logger.info(f"Analyzing file: {file.filename}")
        
        # Extract audio features
        audio_features = audio_processor.extract_all_features(temp_file_path)
        
        # Get transcript
        transcript = audio_features.get("transcript", "")
        
        # Extract text features if transcript available
        text_features = {}
        if transcript:
            text_features = text_processor.extract_all_features(transcript)
        
        # Combine all features
        all_features = {
            "file_path": temp_file_path,
            "transcript": transcript,
            **audio_features,
            **text_features
        }
        
        # Convert to DataFrame for model
        df = pd.DataFrame([all_features])
        
        # Check if model needs training (if it's the first run)
        if not hasattr(model, 'combined_pipeline') or model.combined_pipeline is None:
            logger.info("No trained model found. Training on current data.")
            model.train(df)
        
        # Get risk score
        prediction_result = model.predict(df)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])
        
        # Extract the prediction for the single sample
        # Handle both normal predictions and demo mode results
        if "predictions" in prediction_result:
            prediction = prediction_result["predictions"][0]
        elif "risk_scores" in prediction_result:
            # Demo mode format
            prediction = {
                "risk_score": prediction_result["risk_scores"][0],
                "risk_level": prediction_result["risk_levels"][0],
                "anomaly_score": prediction_result["anomaly_scores"][0] if "anomaly_scores" in prediction_result else 0.5
            }
        
        # Get key indicators
        key_indicators = []
        
        # Handle different formats of feature importance (normal mode vs demo mode)
        if "feature_importance" in prediction_result and isinstance(prediction_result["feature_importance"], list):
            # Direct format from demo mode
            for item in prediction_result["feature_importance"]:
                key_indicators.append({
                    "feature": item["feature"],
                    "importance": item["importance"],
                    "value": audio_features.get(item["feature"], 0)
                })
        else:
            # Normal format from trained model
            for feature, importance in model.get_feature_importance().items():
                if feature in audio_features or feature in text_features:
                    feature_value = audio_features.get(feature, text_features.get(feature, 0))
                    key_indicators.append({
                        "feature": feature,
                        "importance": importance,
                        "value": feature_value
                    })
        
        # Sort by importance and get top 5
        key_indicators = sorted(key_indicators, key=lambda x: x["importance"], reverse=True)[:5]
        
        # Clean up the temporary file
        clean_up_file(temp_file_path)
        
        # Return results
        return {
            "risk_score": prediction["risk_score"],
            "risk_level": prediction["risk_level"],
            "transcript": transcript,
            "key_indicators": key_indicators,
            "audio_features": {k: v for k, v in audio_features.items() 
                              if k not in ["file_path", "transcript"]},
            "text_features": text_features
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-train")
async def batch_train(files: List[UploadFile] = File(...)):
    """
    Train the model on a batch of audio files.
    
    - **files**: List of audio files (WAV format recommended)
    
    Returns training results.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        # Save all uploaded files
        temp_file_paths = []
        for file in files:
            temp_file_path = save_uploaded_file(file.file)
            temp_file_paths.append(temp_file_path)
        
        logger.info(f"Training on {len(temp_file_paths)} files")
        
        # Process all files
        all_features = []
        for file_path in temp_file_paths:
            # Extract audio features
            audio_features = audio_processor.extract_all_features(file_path)
            
            # Get transcript
            transcript = audio_features.get("transcript", "")
            
            # Extract text features if transcript available
            text_features = {}
            if transcript:
                text_features = text_processor.extract_all_features(transcript)
            
            # Combine all features
            combined_features = {
                "file_path": file_path,
                "transcript": transcript,
                **audio_features,
                **text_features
            }
            
            all_features.append(combined_features)
        
        # Convert to DataFrame for model
        df = pd.DataFrame(all_features)
        
        # Train model
        training_results = model.train(df)
        
        # Clean up temporary files
        for file_path in temp_file_paths:
            clean_up_file(file_path)
        
        return {"message": "Model trained successfully", "details": training_results}
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
