"""
Test script for the Voice Detection API
Run with: python -m pytest tests/test_api.py -v
"""
import base64
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "sk_hackathon_voice_detect_2024"

# Sample Base64 audio (minimal valid MP3 header for testing)
# This is a minimal MP3 file that will trigger the processing pipeline
SAMPLE_AUDIO_BASE64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAABhgC7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7//////////////////////////////////////////////////////////////////8AAAAATGF2YzU2LjQxAAAAAAAAAAAAAAAAJAAAAAAAAAAAAYYNAAAAAAAAAAAAAAAAAAAA//tQZAAP8AAAaQAAAAgAAA0gAAABAAABpAAAACAAADSAAAAETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tQZB4P8AAAaQAAAAgAAA0gAAABAAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQ=="


class TestHealthCheck:
    """Test the health check endpoint."""
    
    def test_health_check(self):
        """Test that health check returns correct status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "languages" in data


class TestAuthentication:
    """Test API key authentication."""
    
    def test_missing_api_key(self):
        """Test that missing API key returns 401."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        assert response.status_code == 401
    
    def test_invalid_api_key(self):
        """Test that invalid API key returns 401."""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": "invalid_key"},
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        assert response.status_code == 401
    
    def test_valid_api_key(self):
        """Test that valid API key is accepted."""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": TEST_API_KEY},
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        # May return 400 due to invalid audio, but not 401
        assert response.status_code != 401


class TestRequestValidation:
    """Test request body validation."""
    
    def test_invalid_language(self):
        """Test that invalid language is rejected."""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": TEST_API_KEY},
            json={
                "language": "French",
                "audioFormat": "mp3",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_format(self):
        """Test that invalid audio format is rejected."""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": TEST_API_KEY},
            json={
                "language": "Tamil",
                "audioFormat": "wav",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_missing_audio(self):
        """Test that missing audio is rejected."""
        response = client.post(
            "/api/voice-detection",
            headers={"x-api-key": TEST_API_KEY},
            json={
                "language": "Tamil",
                "audioFormat": "mp3"
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_all_supported_languages(self):
        """Test that all supported languages are accepted."""
        languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        for lang in languages:
            response = client.post(
                "/api/voice-detection",
                headers={"x-api-key": TEST_API_KEY},
                json={
                    "language": lang,
                    "audioFormat": "mp3",
                    "audioBase64": SAMPLE_AUDIO_BASE64
                }
            )
            # Should not get validation error for language
            assert response.status_code != 422 or "language" not in str(response.json())


class TestResponseFormat:
    """Test response format compliance."""
    
    def test_error_response_format(self):
        """Test that error responses have correct format."""
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": SAMPLE_AUDIO_BASE64
            }
        )
        data = response.json()
        # Should have status and message for error
        assert "detail" in data or "status" in data


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
