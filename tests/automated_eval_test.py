#!/usr/bin/env python3
"""
Automated Evaluation Test Script
Simulates how judges might test your API during the 2-hour event

Run: python tests/automated_eval_test.py
"""

import requests
import base64
import json
import time
import os
from pathlib import Path

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key-12345")

# Test results
results = {
    "passed": 0,
    "failed": 0,
    "total_time": 0,
    "tests": []
}


def test_api_health():
    """Test 1: API is running and healthy"""
    print("🧪 Test 1: API Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ PASSED - API is healthy")
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)}")
        return False


def test_single_detection(audio_path: str, expected_result: str, test_name: str):
    """Test single audio detection"""
    print(f"🧪 {test_name}")
    
    try:
        # Load and encode audio
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Make request
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "language": "English",
                "audioBase64": audio_base64
            },
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            classification = data.get("classification", "UNKNOWN")
            confidence = data.get("confidenceScore", 0)
            
            passed = classification == expected_result
            status = "✅ PASSED" if passed else "❌ FAILED"
            
            print(f"   {status}")
            print(f"   Expected: {expected_result} | Got: {classification} ({confidence:.0%})")
            print(f"   Time: {elapsed:.2f}s")
            
            results["tests"].append({
                "name": test_name,
                "passed": passed,
                "expected": expected_result,
                "actual": classification,
                "confidence": confidence,
                "time": elapsed
            })
            results["total_time"] += elapsed
            results["passed" if passed else "failed"] += 1
            return passed
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            results["failed"] += 1
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)}")
        results["failed"] += 1
        return False


def test_batch_processing():
    """Test batch processing endpoint"""
    print("🧪 Test: Batch Processing")
    
    # This would require sample files - skip if not available
    print("   ⏭️ SKIPPED - No batch test files configured")
    return True


def test_edge_cases():
    """Test edge case handling"""
    print("\n🧪 Edge Case Tests")
    
    # Test 1: Empty audio
    print("   Testing empty request...")
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "language": "English",
                "audioBase64": ""
            },
            timeout=10
        )
        if response.status_code == 400:
            print("   ✅ Empty audio handled correctly (400 error)")
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
    except:
        print("   ❌ Error handling empty request")
    
    # Test 2: Invalid base64
    print("   Testing invalid base64...")
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "language": "English",
                "audioBase64": "not-valid-base64!!!"
            },
            timeout=10
        )
        if response.status_code in [400, 422]:
            print("   ✅ Invalid base64 handled correctly")
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
    except:
        print("   ❌ Error handling invalid base64")
    
    # Test 3: Unsupported language
    print("   Testing unsupported language...")
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "language": "Martian",
                "audioBase64": "SGVsbG8gV29ybGQ="  # "Hello World" in base64
            },
            timeout=10
        )
        if response.status_code in [400, 422]:
            print("   ✅ Unsupported language handled correctly")
        else:
            print(f"   ⚠️ Status: {response.status_code} (may still work)")
    except:
        print("   ❌ Error handling unsupported language")


def test_response_time():
    """Test response time requirements"""
    print("\n🧪 Response Time Analysis")
    
    if results["tests"]:
        times = [t["time"] for t in results["tests"]]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Min: {min_time:.2f}s | Max: {max_time:.2f}s")
        
        if avg_time < 5:
            print("   ✅ Response time is good (< 5s average)")
        elif avg_time < 10:
            print("   ⚠️ Response time is acceptable (< 10s average)")
        else:
            print("   ❌ Response time is slow (> 10s average)")
    else:
        print("   ⏭️ No timing data available")


def print_summary():
    """Print final summary"""
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    
    total = results["passed"] + results["failed"]
    if total > 0:
        accuracy = results["passed"] / total * 100
        print(f"   Tests Passed: {results['passed']}/{total} ({accuracy:.0f}%)")
        print(f"   Total Time: {results['total_time']:.2f}s")
        
        if accuracy >= 90:
            print("\n   🏆 EXCELLENT - Ready for evaluation!")
        elif accuracy >= 75:
            print("\n   ✅ GOOD - Minor improvements needed")
        elif accuracy >= 50:
            print("\n   ⚠️ NEEDS WORK - Review failed tests")
        else:
            print("\n   ❌ CRITICAL - Major issues to fix")
    else:
        print("   No tests were run")
    
    print("="*50)


def main():
    print("="*50)
    print("🚀 AI Voice Detection - Automated Evaluation Test")
    print("="*50)
    print(f"API URL: {API_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Run tests
    test_api_health()
    
    # Check for sample audio files
    sample_dir = Path("tests/samples")
    if sample_dir.exists():
        print("\n📁 Found sample audio files")
        for audio_file in sample_dir.glob("*.wav"):
            # Determine expected result from filename
            if "ai" in audio_file.name.lower() or "synthetic" in audio_file.name.lower():
                expected = "AI_GENERATED"
            else:
                expected = "HUMAN"
            test_single_detection(str(audio_file), expected, f"Test: {audio_file.name}")
    else:
        print("\n⚠️ No sample files found in tests/samples/")
        print("   Create this folder with test audio files for full testing")
    
    test_edge_cases()
    test_response_time()
    print_summary()


if __name__ == "__main__":
    main()
