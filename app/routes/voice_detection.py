"""
Voice Detection API Routes
Handles the main voice detection endpoint
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import hashlib
import time

from fastapi import APIRouter, Request, Depends, HTTPException

from app.middleware.auth import verify_api_key
from app.audio_processor import audio_processor
from app.voice_detector import voice_detector


# Create router
router = APIRouter(prefix="/api", tags=["Voice Detection"])


# Simple in-memory cache for repeated requests (TTL: 5 minutes)
_detection_cache: Dict[str, Dict[str, Any]] = {}
_cache_ttl = 300  # 5 minutes


def _get_cache_key(audio_b64: str) -> str:
    """Generate cache key from audio hash (first/last 1000 chars + length)"""
    content = f"{audio_b64[:1000]}_{audio_b64[-1000:]}_{len(audio_b64)}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get result from cache if valid"""
    if cache_key in _detection_cache:
        entry = _detection_cache[cache_key]
        if time.time() - entry['timestamp'] < _cache_ttl:
            return entry['result']
        else:
            del _detection_cache[cache_key]
    return None


def _cache_result(cache_key: str, result: Dict[str, Any]):
    """Store result in cache"""
    # Limit cache size to 100 entries
    if len(_detection_cache) > 100:
        # Remove oldest entry
        oldest = min(_detection_cache.keys(), key=lambda k: _detection_cache[k]['timestamp'])
        del _detection_cache[oldest]
    
    _detection_cache[cache_key] = {
        'timestamp': time.time(),
        'result': result
    }


# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection endpoint."""
    
    language: str = Field(
        ..., 
        description="Language of the audio. Must be one of: Tamil, English, Hindi, Malayalam, Telugu"
    )
    audioFormat: str = Field(
        ..., 
        description="Format of the audio. Must be 'mp3'"
    )
    audioBase64: str = Field(
        ..., 
        description="Base64-encoded MP3 audio data"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
        return v
    
    @field_validator('audioFormat')
    @classmethod
    def validate_format(cls, v):
        if v.lower() != 'mp3':
            raise ValueError("audioFormat must be 'mp3'")
        return v.lower()
    
    @field_validator('audioBase64')
    @classmethod
    def validate_audio(cls, v):
        if not v or len(v) < 100:
            raise ValueError("audioBase64 is required and must contain valid Base64 audio data")
        return v


class ModelScores(BaseModel):
    """Individual model scores breakdown."""
    heuristic: Optional[float] = Field(None, description="Heuristic analysis score")
    cnn_mfcc: Optional[float] = Field(None, description="CNN MFCC model score")
    transformers: Optional[float] = Field(None, description="Wav2Vec2 transformer score")


class VoiceDetectionResponse(BaseModel):
    """Response body for successful voice detection."""
    
    status: str = Field(default="success", description="Response status")
    language: str = Field(..., description="Language of the analyzed audio")
    classification: str = Field(
        ..., 
        description="Classification result: 'AI_GENERATED' or 'HUMAN'"
    )
    confidenceScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ..., 
        description="Short explanation for the classification decision"
    )
    modelScores: Optional[ModelScores] = Field(
        None,
        description="Individual scores from each detection model"
    )
    spectrogramBase64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG image of the Mel-Spectrogram for visual verification"
    )


class ErrorResponse(BaseModel):
    """Response body for errors."""
    
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse, "description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-Generated Voice",
    description="Analyzes an MP3 audio sample and classifies it as AI-generated or human-spoken."
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect whether a voice sample is AI-generated or human-spoken.
    
    This endpoint accepts a Base64-encoded MP3 audio file and returns
    the classification result along with a confidence score.
    
    **Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu
    """
    try:
        # Check cache first for faster response on repeated requests
        cache_key = _get_cache_key(request.audioBase64)
        cached = _get_cached_result(cache_key)
        if cached:
            # Return cached result with updated language
            cached['language'] = request.language
            return VoiceDetectionResponse(**cached)
        
        # Decode audio bytes for HF API
        import base64
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Validate audio length (minimum 0.5 seconds)
        if len(audio_bytes) < 8000:  # Rough minimum for 0.5s audio
            raise ValueError("Audio too short. Minimum 0.5 seconds required.")
        
        # Process audio and extract features + raw samples
        features, audio_samples, sample_rate = audio_processor.process_audio_with_samples(
            request.audioBase64
        )
        
        # Validate audio duration
        duration = len(audio_samples) / sample_rate if sample_rate > 0 else 0
        if duration < 0.5:
            raise ValueError(f"Audio too short ({duration:.1f}s). Minimum 0.5 seconds required.")
        
        # Detect voice type using ensemble (heuristic + local ML + HF API)
        result = voice_detector.detect(
            features, 
            audio=audio_samples, 
            sr=sample_rate,
            audio_bytes=audio_bytes
        )
        
        # Generate visual explanation (Spectrogram)
        spec_b64 = audio_processor.generate_spectrogram_base64(audio_samples, sample_rate)

        # Extract model scores if available
        model_scores = None
        if 'model_scores' in result:
            model_scores = ModelScores(
                heuristic=result['model_scores'].get('heuristic'),
                cnn_mfcc=result['model_scores'].get('cnn_mfcc'),
                transformers=result['model_scores'].get('transformers')
            )
        
        # Build response
        response_data = {
            "status": "success",
            "language": request.language,
            "classification": result['classification'],
            "confidenceScore": result['confidenceScore'],
            "explanation": result['explanation'],
            "modelScores": model_scores,
            "spectrogramBase64": spec_b64
        }
        
        # Cache the result for faster repeat requests
        _cache_result(cache_key, response_data)
        
        return VoiceDetectionResponse(**response_data)
        
    except ValueError as e:
        # Handle audio processing errors
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Audio processing error: {str(e)}"
            }
        )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )


# Batch Processing Models
class BatchAudioItem(BaseModel):
    """Single audio item in batch request."""
    id: str = Field(..., description="Unique identifier for this audio sample")
    language: str = Field(..., description="Language of the audio")
    audioFormat: str = Field(default="mp3", description="Audio format")
    audioBase64: str = Field(..., description="Base64 encoded audio")


class BatchRequest(BaseModel):
    """Request body for batch voice detection."""
    items: List[BatchAudioItem] = Field(..., description="List of audio items to process", max_length=10)


class BatchResultItem(BaseModel):
    """Single result in batch response."""
    id: str
    status: str
    classification: Optional[str] = None
    confidenceScore: Optional[float] = None
    explanation: Optional[str] = None
    modelScores: Optional[ModelScores] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Response body for batch voice detection."""
    status: str = "success"
    totalItems: int
    successCount: int
    errorCount: int
    results: List[BatchResultItem]


@router.post(
    "/voice-detection/batch",
    response_model=BatchResponse,
    summary="Batch Voice Detection",
    description="Process multiple audio samples in a single request. Max 10 items per batch."
)
async def detect_voice_batch(
    request: BatchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Process multiple audio samples for voice detection.
    
    **Limits:**
    - Maximum 10 audio samples per batch
    - No spectrograms returned in batch mode (for performance)
    """
    import base64
    
    results = []
    success_count = 0
    error_count = 0
    
    for item in request.items:
        try:
            # Decode and process audio
            audio_bytes = base64.b64decode(item.audioBase64)
            features, audio_samples, sample_rate = audio_processor.process_audio_with_samples(
                item.audioBase64
            )
            
            # Detect
            result = voice_detector.detect(
                features,
                audio=audio_samples,
                sr=sample_rate,
                audio_bytes=audio_bytes
            )
            
            # Extract model scores
            model_scores = None
            if 'model_scores' in result:
                model_scores = ModelScores(
                    heuristic=result['model_scores'].get('heuristic'),
                    cnn_mfcc=result['model_scores'].get('cnn_mfcc'),
                    transformers=result['model_scores'].get('transformers')
                )
            
            results.append(BatchResultItem(
                id=item.id,
                status="success",
                classification=result['classification'],
                confidenceScore=result['confidenceScore'],
                explanation=result['explanation'],
                modelScores=model_scores
            ))
            success_count += 1
            
        except Exception as e:
            results.append(BatchResultItem(
                id=item.id,
                status="error",
                error=str(e)
            ))
            error_count += 1
    
    return BatchResponse(
        status="success",
        totalItems=len(request.items),
        successCount=success_count,
        errorCount=error_count,
        results=results
    )

