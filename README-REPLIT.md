# MemoTag Cognitive Decline Detection API

## Overview
This API analyzes voice recordings to detect potential indicators of cognitive decline by examining speech patterns, hesitations, pauses, and linguistic features.

## API Endpoints

### `/health`
Check if the API is running properly.
- **Method:** GET
- **Response:** `{"status": "healthy"}`

### `/analyze`
Analyze an audio file for cognitive decline indicators.
- **Method:** POST
- **Request:** Form data with a file field containing an audio file (WAV, MP3, FLAC, etc.)
- **Response:** 
  ```json
  {
    "risk_score": 60.0,
    "risk_level": "Moderate",
    "transcript": "Sample transcript...",
    "key_indicators": [
      {"feature": "pause_patterns", "importance": 0.35, "value": 4},
      {"feature": "speech_rhythm", "importance": 0.25, "value": 0.8}
    ],
    "audio_features": {...},
    "text_features": {...}
  }
  ```

### `/batch-train`
Train the model using a batch of audio files.
- **Method:** POST
- **Request:** Form data with multiple files
- **Response:** Training results

## How to Test

### Using cURL
```bash
curl -X POST https://memotag-cognitive-api.yourreplit.repl.co/analyze \
  -F "file=@/path/to/your/audio.wav"
```

### Using Python
```python
import requests

url = "https://memotag-cognitive-api.yourreplit.repl.co/analyze"
files = {'file': open('audio.wav', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### Using Postman
1. Create a POST request to the API URL
2. Go to the "Body" tab
3. Select "form-data"
4. Add a key named "file", set type to "File"
5. Upload your audio file
6. Send the request

## Indicators of Cognitive Decline
- Frequent pauses
- Hesitation markers ("um", "uh")
- Reduced speech rate
- Lower pitch variability
- Word-finding difficulties
- Simplified sentence structure

## Features Extracted
- Acoustic features (pauses, pitch, rhythm, etc.)
- Linguistic features (hesitation markers, sentence complexity, etc.)
- Emotional content
- Speech coherence
