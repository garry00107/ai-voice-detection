"""
AI Voice Detection API

Main FastAPI application entry point for AI-generated voice detection.
Provides REST API endpoints for analyzing audio samples to determine
whether they contain AI-generated or genuine human speech.

Supported languages: Tamil, English, Hindi, Malayalam, Telugu

Author: Gaurav Sulsule
Built for: India AI Impact Buildathon 2026
"""
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from contextlib import asynccontextmanager

from app.routes.voice_detection import router as voice_router
from app.ml_detector import get_ml_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup: Pre-loads ML models for faster inference.
    Shutdown: Cleans up resources gracefully.
    """
    logger.info("Starting up... Pre-loading ML models")
    try:
        detector = get_ml_detector()
        detector.load_model()
        logger.info("ML Models loaded successfully")
    except Exception as e:
        logger.warning(f"Model loading failed (will lazy-load on first request): {e}")
    
    yield
    
    logger.info("Shutting down gracefully...")

# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    lifespan=lifespan,
    description="""
    REST API for detecting AI-generated voices in audio samples.
    
    ## Features
    - Detects AI-generated vs human voices
    - Supports 5 languages: Tamil, English, Hindi, Malayalam, Telugu
    - Returns confidence scores and explanations
    - Secure API key authentication
    
    ## Usage
    Send a POST request to `/api/voice-detection` with:
    - `x-api-key` header for authentication
    - JSON body with `language`, `audioFormat`, and `audioBase64`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(voice_router)


# Health check moved to /health endpoint to avoid conflict with Gradio UI redirect


@app.get("/health", tags=["Health"])
async def health_endpoint():
    """
    Health check endpoint for automated evaluation.
    Returns API status, version, and model loading state.
    """
    from app.ml_detector import get_ml_detector
    
    try:
        detector = get_ml_detector()
        models_loaded = detector.model is not None
    except:
        models_loaded = False
    
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "endpoints": {
            "single": "/api/voice-detection",
            "batch": "/api/voice-detection/batch",
            "docs": "/docs"
        }
    }


# Root POST endpoint for hackathon compatibility
from app.audio_processor import audio_processor
from app.voice_detector import voice_detector
from pydantic import BaseModel
import base64
from fastapi.responses import RedirectResponse

class HackathonRequest(BaseModel):
    language: str = "english"
    audioFormat: str = "mp3"
    audioBase64: str

# Gradio UI is now mounted at root path directly in gradio_app.py

@app.post("/", tags=["Detection"])
async def root_detect(request: HackathonRequest):
    """
    Root detection endpoint for hackathon.
    Accepts POST with audioBase64, language, audioFormat.
    """
    import datetime
    import traceback
    import numpy as np
    
    request_id = datetime.datetime.now().strftime("%H%M%S%f")[:10]
    start_time = datetime.datetime.now()
    
    # ======= LOGGING: Request received =======
    print(f"\n{'='*70}")
    print(f"📥 REQUEST #{request_id} | {start_time.isoformat()}")
    print(f"   Language: {request.language} | Format: {request.audioFormat}")
    
    try:
        # ======= Step 1: Validate and decode base64 =======
        if not request.audioBase64:
            print(f"   ❌ Empty audioBase64 received")
            return {
                "status": "error",
                "message": "audioBase64 is empty"
            }
        
        base64_len = len(request.audioBase64)
        print(f"   Base64 length: {base64_len:,} chars")
        
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception as decode_err:
            print(f"   ❌ Base64 decode failed: {decode_err}")
            return {
                "status": "error", 
                "message": f"Invalid base64 encoding: {str(decode_err)}"
            }
        
        audio_bytes_len = len(audio_bytes)
        print(f"   Decoded: {audio_bytes_len:,} bytes ({audio_bytes_len/1024:.1f} KB)")
        
        if audio_bytes_len < 100:
            print(f"   ⚠️ Audio too short ({audio_bytes_len} bytes)")
            return {
                "status": "success",
                "language": request.language,
                "classification": "HUMAN",  # Default to HUMAN for tiny samples
                "confidenceScore": 0.5,
                "explanation": "Audio sample too short for reliable detection"
            }
        
        # ======= Step 2: Process audio =======
        try:
            features, audio_samples, sample_rate = audio_processor.process_audio_with_samples(
                request.audioBase64
            )
        except Exception as proc_err:
            print(f"   ❌ Audio processing failed: {proc_err}")
            # Fallback: try detection with just raw bytes
            try:
                result = voice_detector.detect(
                    {}, audio=None, sr=None, audio_bytes=audio_bytes
                )
                return {
                    "status": "success",
                    "language": request.language,
                    "classification": result['classification'],
                    "confidenceScore": result['confidenceScore'],
                    "explanation": "Processed with transformers only (audio decode fallback)"
                }
            except:
                return {
                    "status": "error",
                    "message": f"Audio processing failed: {str(proc_err)}"
                }
        
        # ======= LOGGING: Audio characteristics =======
        if audio_samples is not None and len(audio_samples) > 0:
            duration = len(audio_samples) / sample_rate
            audio_rms = float(np.sqrt(np.mean(audio_samples**2)))
            audio_max = float(np.max(np.abs(audio_samples)))
            
            print(f"   Duration: {duration:.2f}s | SR: {sample_rate}Hz | Samples: {len(audio_samples):,}")
            print(f"   RMS: {audio_rms:.4f} | Peak: {audio_max:.4f}")
        else:
            duration = 0
            print(f"   ⚠️ No audio samples extracted, using transformers only")
        
        # ======= Step 3: Run detection =======
        try:
            result = voice_detector.detect(
                features, 
                audio=audio_samples, 
                sr=sample_rate,
                audio_bytes=audio_bytes
            )
        except Exception as detect_err:
            print(f"   ❌ Detection failed: {detect_err}")
            traceback.print_exc()
            # Return safe default
            return {
                "status": "success",
                "language": request.language,
                "classification": "HUMAN",
                "confidenceScore": 0.5,
                "explanation": "Detection error, defaulting to HUMAN"
            }
        
        # ======= LOGGING: Detection results =======
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n🔍 RESULT #{request_id}:")
        print(f"   Classification: {result['classification']} | Confidence: {result['confidenceScore']:.3f}")
        print(f"   Method: {result.get('method', 'unknown')}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "language": request.language,
            "classification": result['classification'],
            "confidenceScore": result['confidenceScore'],
            "explanation": result['explanation']
        }
        
    except Exception as e:
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n❌ ERROR #{request_id}: {str(e)}")
        print(f"   Time: {elapsed:.2f}s")
        traceback.print_exc()
        print(f"{'='*70}\n")
        
        # Return safe default on any unhandled error
        return {
            "status": "success",
            "language": request.language,
            "classification": "HUMAN",
            "confidenceScore": 0.5,
            "explanation": "Processing error, defaulting to HUMAN"
        }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error. Please try again later."
        }
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
