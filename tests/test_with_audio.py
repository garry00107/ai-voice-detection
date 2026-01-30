#!/usr/bin/env python3
"""
Test script to verify the voice detection API with sample audio.
This generates a synthetic test audio and sends it to the API.
"""
import base64
import json
import sys
import os
import subprocess
import tempfile

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_audio():
    """Create a simple test MP3 file using FFmpeg (if available) or return sample bytes."""
    try:
        # Try to create a simple audio using FFmpeg (generates a 1-second sine wave)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            temp_path = f.name
        
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 
            'sine=frequency=440:duration=1',
            '-acodec', 'libmp3lame', '-b:a', '64k',
            temp_path
        ], capture_output=True)
        
        if result.returncode == 0:
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(temp_path)
            return audio_bytes
    except FileNotFoundError:
        pass
    
    # Fallback: Return a minimal valid MP3 header + frame
    # This is a minimal MP3 file that may not play but should be processable
    return None

def test_api():
    """Test the voice detection API."""
    import requests
    
    API_URL = "http://127.0.0.1:8000/api/voice-detection"
    API_KEY = "sk_hackathon_voice_detect_2024"
    
    # Try to create test audio
    audio_bytes = create_test_audio()
    
    if audio_bytes is None:
        print("Could not create test audio. Skipping audio test.")
        return
    
    # Encode to base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Test request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
    
    print(f"Testing API with {len(audio_bytes)} byte audio file...")
    print(f"Base64 length: {len(audio_base64)}")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ API test PASSED!")
        else:
            print("\n❌ API test FAILED!")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()
