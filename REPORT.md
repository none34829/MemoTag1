# MemoTag Voice-Based Cognitive Decline Detection
## Analysis Report & Implementation Details

## 1. Executive Summary

This project implements a proof-of-concept pipeline for detecting early indicators of cognitive decline from voice data. The system analyzes both acoustic features of speech and linguistic patterns in transcribed text to identify potential cognitive impairment markers.

Key components include:
- Audio feature extraction (pauses, hesitations, pitch, rhythm)
- Linguistic analysis (word-finding difficulties, syntactic complexity)
- Unsupervised machine learning for pattern detection
- API endpoint for remote testing and integration

## 2. Feature Engineering Approach

### 2.1 Audio Features

We extracted several categories of acoustic features that research has shown correlate with cognitive decline:

| Feature Category | Specific Metrics | Cognitive Relevance |
|------------------|------------------|---------------------|
| **Pauses & Timing** | Pause count, average pause duration, pause rate | Increased pauses often indicate word retrieval issues |
| **Pitch Patterns** | Pitch mean, standard deviation, variability coefficient | Reduced pitch variation suggests flat affect, common in cognitive decline |
| **Rhythm Features** | Tempo, zero-crossing rate, rhythm regularity | Disrupted rhythm patterns correlate with executive function issues |
| **Spectral Features** | Spectral centroid, bandwidth, flatness | Captures subtle changes in voice timbre and articulation |

### 2.2 Linguistic Features

From the transcribed speech, we analyzed:

| Feature Category | Specific Metrics | Cognitive Relevance |
|------------------|------------------|---------------------|
| **Hesitation Markers** | Count and ratio of fillers (um, uh, etc.) | Higher usage indicates word-finding difficulties |
| **Word-Finding Difficulties** | Phrases indicating lexical access problems | Direct indicators of memory retrieval issues |
| **Sentence Complexity** | Avg. sentence length, syntactic complexity | Simplified syntax often appears with cognitive decline |
| **Discourse Coherence** | Pronoun-noun ratio, topic consistency | Measures ability to maintain coherent narrative |

## 3. Machine Learning Methodology

We implemented an unsupervised approach for this proof-of-concept, as it allows pattern detection without requiring labeled data, which is often difficult to obtain in healthcare applications.

### 3.1 Feature Processing

- **Robust Scaling**: Applied to normalize features while handling outliers
- **Dimensionality Reduction**: PCA to capture 95% of variance, reducing noise and redundancy
- **Feature Importance**: Computed based on PCA component loadings and explained variance

### 3.2 Pattern Detection Techniques

1. **Anomaly Detection (Isolation Forest)**
   - Identifies unusual speech patterns that deviate from the norm
   - Computes anomaly scores for each sample
   - Converted to risk scores (0-100 scale) for interpretability

2. **Clustering (K-means)**
   - Groups similar speech patterns together
   - Optimal cluster count determined via silhouette score
   - Provides insight into natural groupings of cognitive patterns

3. **Feature Correlation Analysis**
   - Identifies which features most strongly correlate with risk scores
   - Helps prioritize the most informative indicators

## 4. Implementation Details

The system is implemented as a Python-based pipeline with the following components:

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

### 4.1 Key Dependencies

- **Audio Processing**: librosa, pydub, SpeechRecognition
- **Speech Recognition**: Whisper, Google Speech API (fallback)
- **NLP**: spaCy, NLTK
- **Machine Learning**: scikit-learn
- **API**: FastAPI, uvicorn
- **Visualization**: matplotlib, seaborn

### 4.2 API Functionality

The API exposes two main endpoints:

1. `/analyze` - Analyzes a single audio file and returns:
   - Risk score (0-100)
   - Risk level classification (Low/Medium/High)
   - Transcript
   - Key indicators with importance values
   - Extracted features

2. `/batch-train` - Trains the model on multiple audio files:
   - Processes batch of samples
   - Extracts features and trains the model
   - Returns training results

## 5. Key Findings

### 5.1 Most Insightful Features

Based on our feature importance analysis, the most predictive indicators of cognitive decline are:

1. **Pause Rate**: Frequency of pauses normalized by speech duration
2. **Hesitation Ratio**: Proportion of filler words in speech
3. **Syntactic Complexity**: Measure of grammatical structure complexity
4. **Pitch Variability**: Variation in vocal pitch during speech
5. **Word-Finding Difficulty Ratio**: Rate of explicit word-finding problems

### 5.2 Pattern Observations

Our unsupervised approach revealed several interesting patterns:

1. **Temporal Patterns**: Speech timing features (pauses, rhythm) were more discriminative than spectral features
2. **Combined Signal Strength**: The combination of audio and linguistic features provided stronger signal than either alone
3. **Risk Distribution**: Most samples clustered in low-to-medium risk ranges, with outliers showing distinct high-risk characteristics

## 6. Potential Next Steps

To make this system clinically robust, several enhancements would be needed:

### 6.1 Data & Validation

- Collect longitudinal data from individuals across the cognitive spectrum
- Partner with clinical specialists to validate findings against established assessments
- Incorporate demographic factors for personalized baseline comparisons

### 6.2 Technical Enhancements

- Implement supervised models once labeled data is available
- Enhance the speech recognition component for better performance with elderly speech
- Add real-time processing capabilities for immediate feedback
- Incorporate additional modalities (e.g., visual cues from video)

### 6.3 Clinical Integration

- Develop longitudinal tracking to monitor changes over time
- Create interpretable reports for healthcare providers
- Implement privacy-preserving techniques for sensitive health data
- Establish thresholds for clinical intervention recommendations

## 7. Deployment & Usage

The system is deployed as a REST API that can be tested using Postman or any HTTP client:

### Testing with Postman:

1. Start the API server:
   ```
   python run_api.py
   ```

2. Configure a Postman request:
   - POST request to `http://localhost:8000/analyze`
   - Use form-data with key 'file' (type: file)
   - Upload an audio file (WAV format recommended)

3. The response includes:
   ```json
   {
     "risk_score": 45.2,
     "risk_level": "Medium",
     "transcript": "Sample transcribed text...",
     "key_indicators": [
       {"feature": "pause_rate", "importance": 0.32, "value": 0.8},
       {"feature": "hesitation_ratio", "importance": 0.25, "value": 0.12},
       ...
     ],
     "audio_features": {...},
     "text_features": {...}
   }
   ```

### Cloud Deployment:

The system can be deployed to cloud platforms (AWS, GCP, Azure) using the provided deployment script:

```
python deploy.py --platform aws --region us-west-2
```

## 8. Conclusion

This proof-of-concept demonstrates the potential of automated voice analysis for early detection of cognitive decline. The combination of acoustic and linguistic features provides a rich set of indicators that correlate with cognitive function. While further validation is needed for clinical use, this approach shows promise as a non-invasive, accessible screening tool that could enable earlier intervention and improved outcomes.

The system's API-based architecture allows for easy integration into existing healthcare workflows and applications, facilitating broader adoption and testing. With continued development and clinical validation, this technology could become a valuable tool in the early detection and monitoring of cognitive impairment.
