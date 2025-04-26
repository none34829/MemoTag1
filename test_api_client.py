"""
Test client for the MemoTag cognitive decline detection API.
"""

import os
import sys
import requests
import json
import argparse
from pathlib import Path

def test_health(base_url="http://localhost:8000"):
    """Test the API health endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"\n--- Health Check ---")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_analyze_audio(audio_file, base_url="http://localhost:8000"):
    """Test the analyze endpoint with an audio file."""
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return False
    
    try:
        print(f"\n--- Analyzing Audio File: {os.path.basename(audio_file)} ---")
        
        # Prepare the file for upload
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            
            # Send request to the analyze endpoint
            print("Sending request to API...")
            response = requests.post(f"{base_url}/analyze", files=files)
            
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Display key results
                print("\nResults:")
                print(f"- Risk Score: {result['risk_score']:.2f}")
                print(f"- Risk Level: {result['risk_level']}")
                
                # Display transcript if available
                if result.get('transcript'):
                    print(f"\nTranscript:")
                    print(f"{result['transcript'][:100]}...")
                
                # Display key indicators
                print("\nKey Indicators:")
                for indicator in result.get('key_indicators', []):
                    print(f"- {indicator['feature']}: {indicator['importance']:.3f}")
                
                # Display some audio features
                print("\nSelected Audio Features:")
                audio_features = result.get('audio_features', {})
                selected_features = [
                    'pause_count', 'avg_pause_duration', 'pause_rate',
                    'pitch_mean', 'pitch_variability_coefficient'
                ]
                
                for feature in selected_features:
                    if feature in audio_features:
                        print(f"- {feature}: {audio_features[feature]}")
                
                return result
            else:
                print(f"Error: {response.text}")
                return False
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return False

def test_batch_train(audio_files, base_url="http://localhost:8000"):
    """Test the batch-train endpoint with multiple audio files."""
    if not audio_files:
        print("No audio files provided")
        return False
    
    # Check that all files exist
    valid_files = []
    for file_path in audio_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Audio file not found: {file_path}")
    
    if not valid_files:
        print("No valid audio files found")
        return False
    
    try:
        print(f"\n--- Batch Training with {len(valid_files)} Files ---")
        
        # Prepare files for upload
        files = []
        for file_path in valid_files:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                files.append(('files', (os.path.basename(file_path), file_content, 'audio/wav')))
        
        # Send request to the batch-train endpoint
        print("Sending request to API...")
        response = requests.post(f"{base_url}/batch-train", files=files)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nBatch Training Results:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error in batch training: {e}")
        return False

def main():
    """Main function to run API tests."""
    parser = argparse.ArgumentParser(description="Test the MemoTag cognitive decline detection API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--health", action="store_true", help="Test health endpoint")
    parser.add_argument("--analyze", metavar="FILE", help="Test analyze endpoint with a specific audio file")
    parser.add_argument("--batch-train", action="store_true", help="Test batch-train endpoint with all test audio files")
    
    args = parser.parse_args()
    
    # Find test audio files
    data_dir = Path("data/raw")
    audio_files = list(data_dir.glob("*.wav"))
    
    if not audio_files:
        print("No audio files found in data/raw directory")
        return
    
    # Run tests based on arguments
    if args.health or args.test_all:
        test_health(args.url)
    
    if args.analyze:
        test_analyze_audio(args.analyze, args.url)
    elif args.test_all:
        # Test analyze with the first audio file
        test_analyze_audio(str(audio_files[0]), args.url)
    
    if args.batch_train or args.test_all:
        test_batch_train([str(f) for f in audio_files], args.url)
    
    if not (args.health or args.analyze or args.batch_train or args.test_all):
        print("No tests specified. Use --help to see available options.")

if __name__ == "__main__":
    main()
