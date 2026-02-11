# AI Voice Detection - Complete Project Summary

> **Purpose**: This document provides a comprehensive overview of the AI Voice Detection project for any AI agent or developer to understand the entire system from scratch.

---

## 🎯 Project Overview

**What it does**: Detects whether an audio recording contains AI-generated (synthetic) voice or genuine human voice.

**Built for**: India AI Impact Buildathon 2026 - Grand Finale (Feb 16, 2026)

**Live Demo**: https://gaurav00107-ai-voice-detection.hf.space  
**GitHub**: https://github.com/garry00107/ai-voice-detection  
**Branch**: `grand-finale-ui` (latest features)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  Gradio Demo    │  │       FastAPI REST API          │  │
│  │  (gradio_app.py)│  │  (app/routes/voice_detection.py)│  │
│  └────────┬────────┘  └─────────────┬───────────────────┘  │
└───────────┼─────────────────────────┼───────────────────────┘
            │                         │
            ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Audio Processor                          │
│                   (app/audio_processor.py)                  │
│  • Base64 decode → WAV                                      │
│  • Resample to 22050Hz                                      │
│  • Extract MFCC features (13 coefficients)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Voice Detector                          │
│                    (app/voice_detector.py)                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐ │
│  │  Heuristic    │ │   CNN Model   │ │  Wav2Vec2         │ │
│  │  Analysis     │ │   (MFCC)      │ │  Transformer      │ │
│  │  Weight: 30%  │ │  Weight: 35%  │ │  Weight: 60%      │ │
│  └───────────────┘ └───────────────┘ └───────────────────┘ │
│                             │                               │
│              Smart Consensus Voting Algorithm               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Classification  │
                    │ AI_GENERATED or │
                    │     HUMAN       │
                    └─────────────────┘
```

---

## 📁 Key Files & Their Purposes

### Core Detection Logic

| File | Purpose |
|------|---------|
| `app/voice_detector.py` | **Main orchestrator** - combines all 3 models with weighted voting |
| `app/audio_processor.py` | Decodes audio, extracts MFCC features |
| `app/cnn_detector.py` | CNN model trained on MFCC spectrograms |
| `app/heuristics.py` | Statistical analysis (pitch variance, entropy, etc.) |

### User Interfaces

| File | Purpose |
|------|---------|
| `gradio_app.py` | **Gradio demo UI** - Single analysis + Batch processing tabs |
| `app/routes/voice_detection.py` | **REST API** - Single and batch endpoints |
| `app/main.py` | FastAPI app entry point |

### Configuration

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker container for HuggingFace Spaces (runs Gradio) |
| `requirements.txt` | Python dependencies |
| `run_demo_backup.sh` | Local backup demo script |

---

## 🔌 API Reference

### Single Audio Detection
```http
POST /api/voice-detection
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "language": "English",
  "audioBase64": "<base64_encoded_audio>"
}
```

**Response**:
```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.85,
  "explanation": "Multiple synthetic voice markers detected...",
  "modelScores": {
    "heuristic": 0.89,
    "cnn_mfcc": 0.70,
    "transformers": 0.0
  },
  "spectrogramBase64": "<base64_image>"
}
```

### Batch Processing
```http
POST /api/voice-detection/batch
Authorization: Bearer <API_KEY>

{
  "items": [
    {"id": "file1", "language": "English", "audioBase64": "..."},
    {"id": "file2", "language": "Hindi", "audioBase64": "..."}
  ]
}
```

---

## 🧠 Detection Algorithm Deep Dive

### Model 1: Heuristic Analysis (30% weight)
Analyzes statistical features:
- **Pitch variance**: AI voices often have unnaturally consistent pitch
- **Zero-crossing rate**: Different patterns in synthetic vs natural speech
- **Spectral entropy**: AI audio tends to be more "clean" with less randomness
- **Pause patterns**: Synthetic TTS has regular, predictable pauses

### Model 2: CNN on MFCC (35% weight)
- Input: 13 MFCC coefficients extracted from audio
- Architecture: Convolutional layers → MaxPooling → Dense layers
- Trained on: Mixed dataset of human and AI-generated voice samples
- Location: `models/cnn_model_mfcc.h5`

### Model 3: Wav2Vec2 Transformer (60% weight - when confident)
- Pre-trained model: `facebook/wav2vec2-base`
- Fine-tuned for binary classification (AI vs Human)
- Most accurate for detecting modern deepfakes
- Note: May timeout on cold starts; CNN+heuristics provide fallback

### Consensus Logic
```python
# Simplified consensus algorithm
if transformer_confident:
    final = transformer_prediction  # Trust deep learning
elif cnn_strong + heuristic_strong > threshold:
    final = majority_vote
else:
    final = weighted_average
```

---

## 🌐 Supported Languages

- 🇮🇳 Tamil
- 🇮🇳 English  
- 🇮🇳 Hindi
- 🇮🇳 Malayalam
- 🇮🇳 Telugu

---

## 🚀 Running Locally

### Prerequisites
```bash
Python 3.10+
pip install -r requirements.txt
```

### Start Gradio Demo
```bash
cd /Users/garrry/Downloads/Hackathon
source venv/bin/activate
python gradio_app.py
# Opens at http://localhost:7860
```

### Start FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# API at http://localhost:8000/docs
```

---

## 📦 Deployment

### HuggingFace Spaces
- Space ID: `gaurav00107/ai-voice-detection`
- Type: Docker space
- Dockerfile runs `gradio_app.py`
- Auto-rebuilds on file upload via HuggingFace API

### Deploy Command
```python
from huggingface_hub import HfApi
api = HfApi(token='YOUR_TOKEN')
api.upload_file(
    path_or_fileobj='gradio_app.py',
    path_in_repo='gradio_app.py',
    repo_id='gaurav00107/ai-voice-detection',
    repo_type='space'
)
```

---

## 🎨 UI Features

### Single Analysis Tab
- Audio upload (MP3, WAV, OGG) or microphone recording
- Real-time spectrogram visualization (magma colormap)
- Animated confidence meter
- Model scores breakdown
- "How It Works" educational section

### Batch Processing Tab
- Multi-file upload (up to 10 files)
- Summary with AI/Human counts
- Individual file results with color-coding

---

## 🔧 Common Tasks for AI Agents

### Add a New Detection Model
1. Create model file in `app/` (e.g., `app/new_detector.py`)
2. Add model initialization in `app/voice_detector.py`
3. Update consensus logic to include new model's predictions
4. Add weight configuration

### Modify API Response
1. Edit `app/routes/voice_detection.py`
2. Update Pydantic models (`VoiceDetectionResponse`)
3. Test with: `curl -X POST http://localhost:8000/api/voice-detection ...`

### Update Gradio UI
1. Edit `gradio_app.py`
2. Test locally: `python gradio_app.py`
3. Deploy: Upload to HuggingFace via API

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Accuracy | ~85-95% |
| Min Audio Duration | 0.5 seconds |
| Max Batch Size | 10 files |
| Cold Start Time | ~10-15s (model loading) |
| Inference Time | ~2-5s per file |

---

## ⚠️ Known Limitations

1. **Very short audio** (<0.5s) may produce unreliable results
2. **Background noise** can affect accuracy
3. **Transformer model** may timeout on first request (cold start)
4. **Adversarial audio** specifically designed to evade detection not tested

---

## 📞 Contact

**Author**: Gaurav Sulsule  
**GitHub**: github.com/garry00107  
**Event**: India AI Impact Buildathon 2026 Grand Finale
