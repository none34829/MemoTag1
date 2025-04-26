"""
Create test audio files for testing the MemoTag cognitive decline detection API.
"""

import os
import numpy as np
from scipy.io import wavfile
import wave
import struct
from pathlib import Path

def ensure_data_dir():
    """Ensure the data directory exists."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def create_sine_wave(filename, duration=3.0, frequency=440.0, sample_rate=44100):
    """
    Create a simple sine wave audio file.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        frequency: Frequency in Hz
        sample_rate: Sample rate in Hz
    """
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio = (sine_wave * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write(filename, sample_rate, audio)
    print(f"Created sine wave audio file: {filename}")

def create_speech_simulation(filename, duration=5.0, sample_rate=44100):
    """
    Create a simulated speech-like audio file using multiple frequencies.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a speech-like signal with formants and modulation
    # Base frequency (pitch) around 110Hz (typical male) or 220Hz (typical female)
    fundamental = 110
    
    # Add harmonics with varying amplitudes to simulate formants
    signal = 0.5 * np.sin(2 * np.pi * fundamental * t)
    signal += 0.3 * np.sin(2 * np.pi * fundamental * 2 * t)
    signal += 0.15 * np.sin(2 * np.pi * fundamental * 3 * t)
    signal += 0.1 * np.sin(2 * np.pi * fundamental * 4 * t)
    
    # Add some formants (peaks in frequency spectrum characteristic of vowels)
    signal += 0.2 * np.sin(2 * np.pi * 500 * t)  # First formant
    signal += 0.1 * np.sin(2 * np.pi * 1500 * t)  # Second formant
    
    # Amplitude modulation to simulate syllables
    syllable_rate = 4  # syllables per second
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t / 2)
    signal = signal * envelope
    
    # Add a slight fade in and fade out
    fade_samples = int(sample_rate * 0.1)  # 100ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] = signal[:fade_samples] * fade_in
    signal[-fade_samples:] = signal[-fade_samples:] * fade_out
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    audio = (signal * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write(filename, sample_rate, audio)
    print(f"Created simulated speech audio file: {filename}")

def create_speech_with_pauses(filename, duration=8.0, sample_rate=44100):
    """
    Create a simulated speech with pauses to test pause detection.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a speech-like signal with pauses
    signal = np.zeros_like(t)
    
    # Define speech segments and pauses
    segments = [
        (0.0, 2.0),    # Speech
        (2.0, 2.5),    # Pause
        (2.5, 4.5),    # Speech
        (4.5, 5.5),    # Longer pause (hesitation)
        (5.5, 8.0)     # Speech
    ]
    
    fundamental = 120  # fundamental frequency
    
    # Generate signal for each segment
    for start, end in segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment_t = t[start_idx:end_idx] - start  # time relative to segment start
        
        # If this is a speech segment (not a pause)
        if end - start > 0.4:  # Assuming pauses are shorter than 0.4s
            segment_signal = 0.5 * np.sin(2 * np.pi * fundamental * segment_t)
            segment_signal += 0.3 * np.sin(2 * np.pi * fundamental * 2 * segment_t)
            segment_signal += 0.15 * np.sin(2 * np.pi * fundamental * 3 * segment_t)
            
            # Add some formants
            segment_signal += 0.2 * np.sin(2 * np.pi * 500 * segment_t)
            segment_signal += 0.1 * np.sin(2 * np.pi * 1500 * segment_t)
            
            # Amplitude modulation for syllables
            syllable_rate = 3  # syllables per second
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * segment_t / 2)
            segment_signal = segment_signal * envelope
            
            # Add the segment to the main signal
            signal[start_idx:end_idx] = segment_signal
    
    # Add a slight fade in and fade out
    fade_samples = int(sample_rate * 0.1)  # 100ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] = signal[:fade_samples] * fade_in
    signal[-fade_samples:] = signal[-fade_samples:] * fade_out
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    # Convert to 16-bit PCM
    audio = (signal * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write(filename, sample_rate, audio)
    print(f"Created speech with pauses audio file: {filename}")

def create_hesitant_speech(filename, duration=10.0, sample_rate=44100):
    """
    Create a simulated hesitant speech to test cognitive decline detection.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a speech-like signal with pauses, hesitations, and varying pitch
    signal = np.zeros_like(t)
    
    # Define speech segments, hesitations, and pauses
    segments = [
        (0.0, 1.5, 120),     # Normal speech at 120Hz
        (1.5, 2.5, 0),       # Pause (hesitation)
        (2.5, 4.0, 115),     # Slightly lower pitch
        (4.0, 4.3, 0),       # Short pause
        (4.3, 5.8, 125),     # Higher pitch
        (5.8, 7.0, 0),       # Long hesitation
        (7.0, 8.5, 110),     # Lower pitch
        (8.5, 9.0, 0),       # Short pause
        (9.0, 10.0, 118)     # Back to normal-ish
    ]
    
    # Generate signal for each segment
    for start, end, freq in segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment_t = t[start_idx:end_idx] - start  # time relative to segment start
        
        # If this is a speech segment (not a pause)
        if freq > 0:
            segment_signal = 0.5 * np.sin(2 * np.pi * freq * segment_t)
            segment_signal += 0.3 * np.sin(2 * np.pi * freq * 2 * segment_t)
            segment_signal += 0.15 * np.sin(2 * np.pi * freq * 3 * segment_t)
            
            # Add some formants
            segment_signal += 0.2 * np.sin(2 * np.pi * 500 * segment_t)
            segment_signal += 0.1 * np.sin(2 * np.pi * 1500 * segment_t)
            
            # Amplitude modulation for syllables, with variability
            syllable_rate = 2.5 + 1.0 * np.sin(segment_t[0] * 0.5)  # changing rate
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * segment_t / 2)
            segment_signal = segment_signal * envelope
            
            # Add some noise to make it more speech-like
            noise = np.random.normal(0, 0.05, len(segment_t))
            segment_signal += noise
            
            # Add the segment to the main signal
            signal[start_idx:end_idx] = segment_signal
    
    # Add a slight fade in and fade out
    fade_samples = int(sample_rate * 0.1)  # 100ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] = signal[:fade_samples] * fade_in
    signal[-fade_samples:] = signal[-fade_samples:] * fade_out
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    # Convert to 16-bit PCM
    audio = (signal * 32767).astype(np.int16)
    
    # Write to file
    wavfile.write(filename, sample_rate, audio)
    print(f"Created hesitant speech audio file: {filename}")

def main():
    """Create test audio files."""
    data_dir = ensure_data_dir()
    
    # Create different test audio files
    create_sine_wave(os.path.join(data_dir, "test_tone.wav"))
    create_speech_simulation(os.path.join(data_dir, "simulated_speech.wav"))
    create_speech_with_pauses(os.path.join(data_dir, "speech_with_pauses.wav"))
    create_hesitant_speech(os.path.join(data_dir, "hesitant_speech.wav"))
    
    print(f"Created test audio files in {data_dir}")

if __name__ == "__main__":
    main()
