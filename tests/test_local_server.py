import sys
import os
import requests
import base64
import time

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_hackathon_voice_detect_2024"

# Load sample audio
filepath = "sample voice 1.mp3"
if not os.path.exists(filepath):
    print(f"File {filepath} not found!")
    sys.exit(1)

with open(filepath, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": audio_b64
}

print(f"Sending request to {API_URL}...")
try:
    response = requests.post(API_URL, json=payload, headers={"x-api-key": API_KEY})
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Verification Successful!")
        print(f"Classification: {data['classification']}")
        print(f"Confidence: {data['confidenceScore']}")
        print(f"Explanation: {data['explanation']}")
        
        spec = data.get('spectrogramBase64')
        if spec and len(spec) > 100:
            print(f"✅ Spectrogram received! Length: {len(spec)}")
            # Optional: save it to verify image
            with open("test_spectrogram.png", "wb") as imgf:
                imgf.write(base64.b64decode(spec))
            print("   Saved to test_spectrogram.png")
        else:
            print("❌ No spectrogram in response")
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Connection Error: {e}")
    print("Ensure server is running on localhost:8000")
