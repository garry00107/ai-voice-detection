# AI Voice Detection API Documentation
## For Automated Evaluation

---

## Base URL

| Environment | URL |
|-------------|-----|
| Production (HuggingFace) | `https://gaurav00107-ai-voice-detection.hf.space` |
| Local Development | `http://localhost:8000` |

---

## Authentication

**Header:** `Authorization: Bearer <API_KEY>`

For evaluation, use: `test-api-key-12345` (or as provided)

---

## Endpoints

### 1. Single Voice Detection

**POST** `/api/voice-detection`

#### Request

```json
{
  "language": "English",
  "audioBase64": "<base64_encoded_audio>"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `language` | string | Yes | One of: `English`, `Tamil`, `Hindi`, `Malayalam`, `Telugu` |
| `audioBase64` | string | Yes | Base64 encoded audio (WAV, MP3, OGG) |

#### Response (Success - 200)

```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Multiple synthetic voice markers detected...",
  "modelScores": {
    "heuristic": 0.89,
    "cnn_mfcc": 0.70,
    "transformers": 0.92
  },
  "spectrogramBase64": "<base64_encoded_image>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `classification` | string | `AI_GENERATED` or `HUMAN` |
| `confidenceScore` | float | 0.0 to 1.0 |
| `explanation` | string | Human-readable explanation |
| `modelScores` | object | Individual model scores |
| `spectrogramBase64` | string | PNG image in base64 |

#### Response (Error - 400/422)

```json
{
  "status": "error",
  "message": "Audio too short. Minimum 0.5 seconds required."
}
```

---

### 2. Batch Processing

**POST** `/api/voice-detection/batch`

#### Request

```json
{
  "items": [
    {
      "id": "file1",
      "language": "English",
      "audioBase64": "<base64>"
    },
    {
      "id": "file2", 
      "language": "Hindi",
      "audioBase64": "<base64>"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `items` | array | Max 10 items per request |
| `items[].id` | string | Unique identifier for the file |
| `items[].language` | string | Language of the audio |
| `items[].audioBase64` | string | Base64 encoded audio |

#### Response (Success - 200)

```json
{
  "status": "success",
  "results": [
    {
      "id": "file1",
      "classification": "HUMAN",
      "confidenceScore": 0.92
    },
    {
      "id": "file2",
      "classification": "AI_GENERATED",
      "confidenceScore": 0.85
    }
  ],
  "summary": {
    "total": 2,
    "ai_count": 1,
    "human_count": 1
  }
}
```

---

### 3. Health Check

**GET** `/health`

#### Response (200)

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true
}
```

---

## Error Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid audio or parameters |
| 401 | Unauthorized - Invalid API key |
| 422 | Validation Error - Missing required fields |
| 500 | Server Error - Internal processing error |
| 503 | Service Unavailable - Models still loading |

---

## Audio Requirements

| Parameter | Requirement |
|-----------|-------------|
| **Minimum Duration** | 0.5 seconds |
| **Maximum Duration** | 60 seconds |
| **Supported Formats** | WAV, MP3, OGG, FLAC |
| **Sample Rate** | Any (automatically resampled to 22050 Hz) |
| **Channels** | Mono or Stereo (converted to mono) |

---

## Example: cURL

```bash
# Single detection
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key-12345" \
  -d '{
    "language": "English",
    "audioBase64": "'$(base64 -i sample.wav)'"
  }'
```

---

## Example: Python

```python
import requests
import base64

# Load and encode audio
with open("sample.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer test-api-key-12345"
    },
    json={
        "language": "English",
        "audioBase64": audio_b64
    }
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']:.0%}")
```

---

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Batch size | 10 files max |
| Max file size | 10 MB |

---

## Response Time SLA

| Metric | Target |
|--------|--------|
| Average response | < 5 seconds |
| P95 response | < 10 seconds |
| Cold start | < 30 seconds |

---

## Contact

- **GitHub:** github.com/garry00107/ai-voice-detection
- **Demo:** https://gaurav00107-ai-voice-detection.hf.space
