# 🎙️ AI Voice Detection API

> **India AI Impact Buildathon 2026 – Grand Finale Submission**
> **Author:** Gaurav Sulsule  
> **Team:** Team Garry  

A production-grade, multi-lingual REST API that detects AI-generated (deepfake) voices versus natural human speech. The system uses an **ensemble of three independent detection models** to achieve robust, high-confidence classifications across five Indian languages.

![Status](https://img.shields.io/badge/Status-Live-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688)
![Languages](https://img.shields.io/badge/Languages-5-orange)

## 🚀 Live Demo & Deployment

| Resource | URL |
|----------|-----|
| **API Endpoint** | `https://gaurav00107-ai-voice-detection.hf.space/` |
| **Swagger Docs** | [https://gaurav00107-ai-voice-detection.hf.space/docs](https://gaurav00107-ai-voice-detection.hf.space/docs) |
| **Gradio UI** | [https://gaurav00107-ai-voice-detection.hf.space/](https://gaurav00107-ai-voice-detection.hf.space/) |
| **GitHub** | [https://github.com/garry00107/ai-voice-detection](https://github.com/garry00107/ai-voice-detection) |

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              POST / (Root Endpoint)              │    │
│  │  Accepts: { language, audioFormat, audioBase64 } │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │            Audio Processor Module                │    │
│  │  • Base64 decode → MP3 → WAV conversion         │    │
│  │  • Feature extraction (MFCC, pitch, ZCR, etc.)  │    │
│  │  • Resampling to 22050Hz / 16000Hz              │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │          Ensemble Voice Detector                 │    │
│  │                                                  │    │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────────┐  │    │
│  │  │ Heuristic  │ │ CNN (MFCC) │ │ Transformer │  │    │
│  │  │  30% wt    │ │  35% wt    │ │   60% wt    │  │    │
│  │  │            │ │            │ │ Wav2Vec2 +  │  │    │
│  │  │ Pitch, ZCR │ │ 2D Conv    │ │ Deepfake    │  │    │
│  │  │ Spectral   │ │ Layers     │ │ Classifier  │  │    │
│  │  └─────┬──────┘ └─────┬──────┘ └──────┬──────┘  │    │
│  │        └───────────────┼───────────────┘         │    │
│  │                        ▼                         │    │
│  │              Weighted Ensemble Score              │    │
│  │         → Classification + Confidence             │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Returns: { status, classification, confidenceScore }    │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture & Approach

### 1. Heuristic Detector (Weight: 30%)
Statistical analysis of audio features to detect synthetic voice patterns:
- **Pitch Stability**: AI voices exhibit unnaturally consistent pitch (low `pitch_std`)
- **Spectral Entropy**: Synthetic audio has lower spectral randomness
- **Zero-Crossing Rate**: AI voices show more uniform ZCR patterns
- **Spectral Centroid Variation**: Human speech has more natural spectral movement

### 2. CNN MFCC Detector (Weight: 35%)
A custom Convolutional Neural Network trained on MFCC spectrograms:
- **Input**: 13-coefficient MFCC features extracted via Librosa
- **Architecture**: 2D Convolutional layers → BatchNorm → MaxPool → FC layers
- **Training Data**: 2000+ labeled samples from Kaggle human-nonhuman dataset
- **Model File**: `best_model.pt` (PyTorch checkpoint)

### 3. Transformer Deepfake Detector (Weight: 60%)
Fine-tuned Wav2Vec2 model for audio deepfake detection:
- **Base Model**: `facebook/wav2vec2-base` (pre-trained speech representations)
- **Fine-tuned**: `mo-thecreator/Deepfake-audio-detection` (HuggingFace)
- **Input**: Raw 16kHz audio waveform
- **Output**: Binary classification (real vs fake) with probability scores
- **Timeout Protection**: 15-second inference limit to prevent request timeouts

### Ensemble Strategy
The three models vote with weighted scores. The final AI probability is:
```
ensemble_score = (heuristic × 0.30 + cnn × 0.35 + transformer × 0.60) / (sum_of_active_weights)
```
A **consensus override** mechanism detects when heuristic and ML models agree strongly, boosting confidence. If `ensemble_score ≥ 0.5`, the audio is classified as `AI_GENERATED`; otherwise, `HUMAN`.

---

## 🌐 Supported Languages

| Language | Code | Status |
|----------|------|--------|
| English | `en` | ✅ Fully Tested |
| Hindi | `hi` | ✅ Fully Tested |
| Tamil | `ta` | ✅ Fully Tested |
| Malayalam | `ml` | ✅ Fully Tested |
| Telugu | `te` | ✅ Fully Tested |

---

## 🔌 API Documentation

### Authentication
All API requests require an `x-api-key` header:
```
x-api-key: sk_hackathon_voice_detect_2024
```

### Endpoint: `POST /`

**Request Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_AUDIO>"
}
```

**Success Response (200 OK):**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "AI voice detected: synthetic pitch consistency, deep learning model detected synthetic artifacts."
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Description of what went wrong"
}
```

### Additional Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/` | Primary detection endpoint (hackathon) |
| `POST` | `/api/voice-detection` | Full detection with spectrogram |
| `POST` | `/api/voice-detection/batch` | Batch processing (up to 10 files) |
| `GET` | `/health` | Health check and model status |
| `GET` | `/docs` | Interactive Swagger documentation |

---

## 📦 Installation & Local Development

### Prerequisites
- Python 3.10+
- FFmpeg (for audio conversion)
- ~1GB disk space (for ML models)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/garry00107/ai-voice-detection.git
cd ai-voice-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your API_KEY

# 5. Run the server
python gradio_app.py
# Server starts at http://localhost:7860
```

### Docker Deployment
```bash
docker build -t ai-voice-detection .
docker run -p 7860:7860 ai-voice-detection
```

---

## 🧪 Testing

```bash
# Run evaluation against deployed API
python test_evaluation.py

# Run unit tests
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
ai-voice-detection/
├── gradio_app.py              # Main entry point (Gradio UI + FastAPI)
├── app/
│   ├── main.py                # FastAPI application & lifespan management
│   ├── voice_detector.py      # Ensemble detection orchestrator
│   ├── audio_processor.py     # Audio decoding & feature extraction
│   ├── cnn_detector.py        # CNN MFCC model inference
│   ├── ml_detector.py         # Wav2Vec2 ML detector
│   ├── transformers_detector.py  # Transformer deepfake detector
│   ├── middleware/
│   │   └── auth.py            # API key authentication middleware
│   └── routes/
│       └── voice_detection.py # API route handlers & Pydantic models
├── best_model.pt              # Trained CNN model weights
├── tests/                     # Unit and integration tests
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── .env.example               # Environment variable template
└── README.md                  # This file
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Detection Accuracy** | ~92% balanced |
| **AI Voice Detection** | 100% on tested samples |
| **Human Voice Detection** | ~85% on tested samples |
| **Average Response Time** | 5-15 seconds |
| **Max Response Time** | < 30 seconds |
| **Supported Audio Duration** | 0.5s – 60s |

---

## 🔧 Third-Party Libraries & Attribution

| Library | Purpose | License |
|---------|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Web framework | MIT |
| [Gradio](https://gradio.app/) | Demo UI | Apache 2.0 |
| [PyTorch](https://pytorch.org/) | Deep learning framework | BSD |
| [Transformers](https://huggingface.co/docs/transformers) | Pre-trained models | Apache 2.0 |
| [Librosa](https://librosa.org/) | Audio feature extraction | ISC |
| [Scikit-learn](https://scikit-learn.org/) | ML utilities | BSD |
| [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | Speech representations | MIT |
| [mo-thecreator/Deepfake-audio-detection](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) | Deepfake classifier | Apache 2.0 |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the India AI Impact Buildathon 2026 Grand Finale**
