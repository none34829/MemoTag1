"""
Audio processing module for cognitive decline detection.
Extracts acoustic features from voice recordings that may indicate cognitive impairment.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import speech_recognition as sr
import whisper
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Class for processing audio files and extracting cognitive decline indicators."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        # Load whisper model (small is a good balance between accuracy and speed)
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and convert to mono if needed.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logger.info(f"Loading audio file: {file_path}")
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio_data, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def convert_audio_format(self, input_file: str, output_file: str, format: str = "wav") -> str:
        """
        Convert audio to appropriate format for processing.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save converted audio
            format: Target format (default: wav)
            
        Returns:
            Path to converted audio file
        """
        try:
            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format=format)
            logger.info(f"Converted {input_file} to {format} format")
            return output_file
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return input_file  # Return original if conversion fails
    
    def transcribe_audio(self, audio_data: np.ndarray, sr: int) -> str:
        """
        Transcribe audio data to text using Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Transcribed text
        """
        if self.whisper_model is None:
            raise ValueError("Whisper model not loaded")
        
        try:
            # Whisper expects audio in a specific format
            result = self.whisper_model.transcribe(audio_data)
            return result["text"]
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            audio_data, sr = self.load_audio(file_path)
            return self.transcribe_audio(audio_data, sr)
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {e}")
            # Fall back to Google Speech Recognition if Whisper fails
            try:
                with sr.AudioFile(file_path) as source:
                    audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                logger.info("Used fallback Google Speech Recognition")
                return text
            except Exception as e2:
                logger.error(f"Both transcription methods failed: {e2}")
                return ""
    
    def detect_pauses(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect pauses in speech and calculate statistics.
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with pause statistics
        """
        # Convert to AudioSegment for better silence detection
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio_data, sr)
        audio_segment = AudioSegment.from_file(temp_path)
        os.remove(temp_path)
        
        # Detect non-silent chunks
        min_silence_len = 300  # ms
        silence_thresh = -35  # dB
        
        non_silent_ranges = detect_nonsilent(
            audio_segment, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        if not non_silent_ranges:
            return {
                "pause_count": 0,
                "avg_pause_duration": 0,
                "pause_rate": 0,
                "speech_rate": 0,
                "pause_variability": 0
            }
        
        # Calculate pause durations
        pause_durations = []
        speech_durations = []
        
        for i in range(len(non_silent_ranges) - 1):
            curr_end = non_silent_ranges[i][1]
            next_start = non_silent_ranges[i+1][0]
            pause_duration = next_start - curr_end
            
            if pause_duration > min_silence_len:
                pause_durations.append(pause_duration)
            
            speech_durations.append(non_silent_ranges[i][1] - non_silent_ranges[i][0])
        
        # Add the last speech segment
        if non_silent_ranges:
            speech_durations.append(non_silent_ranges[-1][1] - non_silent_ranges[-1][0])
        
        # Calculate statistics
        total_duration = len(audio_segment)
        total_speech_duration = sum(speech_durations)
        
        if not pause_durations:
            avg_pause = 0
            pause_std = 0
        else:
            avg_pause = np.mean(pause_durations)
            pause_std = np.std(pause_durations) if len(pause_durations) > 1 else 0
        
        return {
            "pause_count": len(pause_durations),
            "avg_pause_duration": avg_pause,
            "pause_rate": len(pause_durations) / (total_duration / 1000),  # pauses per second
            "speech_rate": total_speech_duration / total_duration,  # proportion of time speaking
            "pause_variability": pause_std  # standard deviation of pause durations
        }

    def extract_pitch_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pitch-related features from audio.
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with pitch features
        """
        # Extract pitch (F0) using librosa
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        
        # Get the pitch values where magnitude is highest
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Filter out zero pitch values
                pitch_values.append(pitch)
        
        if not pitch_values:
            return {
                "pitch_mean": 0,
                "pitch_std": 0,
                "pitch_range": 0,
                "pitch_variability_coefficient": 0
            }
        
        pitch_array = np.array(pitch_values)
        
        return {
            "pitch_mean": np.mean(pitch_array),
            "pitch_std": np.std(pitch_array),
            "pitch_range": np.max(pitch_array) - np.min(pitch_array),
            "pitch_variability_coefficient": np.std(pitch_array) / np.mean(pitch_array) if np.mean(pitch_array) > 0 else 0
        }
    
    def extract_rhythm_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract rhythm-related features from audio.
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with rhythm features
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # Compute tempo (BPM)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # Rhythm regularity (lower values indicate more regular rhythm)
        if len(onset_env) > 1:
            # Calculate the coefficient of variation in the intervals between onsets
            # Handle different versions of librosa's peak_pick function
            try:
                # Try newer version of peak_pick (which takes a dictionary)
                onset_peaks = librosa.util.peak_pick({
                    'x': onset_env,
                    'pre_max': 3,
                    'post_max': 3,
                    'pre_avg': 3,
                    'post_avg': 5,
                    'delta': 0.5,
                    'wait': 10
                })
            except TypeError:
                try:
                    # Try older version with positional arguments
                    onset_peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
                except Exception as e:
                    # Fallback implementation if both methods fail
                    logger.warning(f"Using fallback peak detection due to error: {e}")
                    onset_peaks = []
                    for i in range(3, len(onset_env) - 3):
                        if onset_env[i] > 0.5 and all(onset_env[i] > onset_env[i-j] for j in range(1, 4)) and all(onset_env[i] > onset_env[i+j] for j in range(1, 4)):
                            onset_peaks.append(i)
            if len(onset_peaks) > 1:
                onset_intervals = np.diff(onset_peaks)
                rhythm_regularity = np.std(onset_intervals) / np.mean(onset_intervals) if np.mean(onset_intervals) > 0 else 0
            else:
                rhythm_regularity = 0
        else:
            rhythm_regularity = 0
        
        return {
            "tempo": tempo,
            "zcr_mean": np.mean(zcr),
            "zcr_std": np.std(zcr),
            "rhythm_regularity": rhythm_regularity
        }
    
    def extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features from audio that may indicate cognitive issues.
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral features
        """
        # Compute spectral centroid (brightness of sound)
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        
        # Compute spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        
        # Compute spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        
        # Compute spectral flatness (1.0 for white noise, 0.0 for pure tone)
        flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Get statistics
        results = {
            "spectral_centroid_mean": np.mean(centroid),
            "spectral_centroid_std": np.std(centroid),
            "spectral_bandwidth_mean": np.mean(bandwidth),
            "spectral_bandwidth_std": np.std(bandwidth),
            "spectral_flatness_mean": np.mean(flatness),
            "spectral_flatness_std": np.std(flatness),
        }
        
        # Add MFCC statistics
        for i in range(13):
            results[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
            results[f"mfcc{i+1}_std"] = np.std(mfccs[i])
        
        return results
    
    def extract_all_features(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all audio features for cognitive decline detection.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info(f"Extracting all features for {file_path}")
        
        try:
            # Load audio
            audio_data, sr = self.load_audio(file_path)
            
            # Get transcript
            transcript = self.transcribe_file(file_path)
            
            # Extract all feature sets
            pause_features = self.detect_pauses(audio_data, sr)
            pitch_features = self.extract_pitch_features(audio_data, sr)
            rhythm_features = self.extract_rhythm_features(audio_data, sr)
            spectral_features = self.extract_spectral_features(audio_data, sr)
            
            # Combine all features
            all_features = {
                "file_path": file_path,
                "transcript": transcript,
                **pause_features,
                **pitch_features,
                **rhythm_features,
                **spectral_features
            }
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            raise

    def batch_process_files(self, file_paths: List[str], output_path: str = None) -> pd.DataFrame:
        """
        Process multiple audio files and compile results into a DataFrame.
        
        Args:
            file_paths: List of paths to audio files
            output_path: Optional path to save results as CSV
            
        Returns:
            DataFrame with features for all files
        """
        all_results = []
        
        for file_path in file_paths:
            try:
                features = self.extract_all_features(file_path)
                all_results.append(features)
                logger.info(f"Processed {file_path} successfully")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
        
        return df
