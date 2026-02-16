"""
Test script to evaluate the AI Voice Detection API 
against the hackathon evaluation criteria.

Uses the provided test samples from 'Hackathon final Samples' folder.
"""
import requests
import base64
import json
import time
import sys

# Configuration
ENDPOINT_URL = "https://gaurav00107-ai-voice-detection.hf.space/"
API_KEY = "sk_hackathon_voice_detect_2024"

# Test files with expected classifications (from filename labels)
TEST_FILES = [
    {
        "language": "English",
        "file_path": "Hackathon final Samples/English_voice_AI_GENERATED.mp3",
        "expected_classification": "AI_GENERATED"
    },
    {
        "language": "Hindi",
        "file_path": "Hackathon final Samples/Hindi_Voice_HUMAN.mp3",
        "expected_classification": "HUMAN"
    },
    {
        "language": "Malayalam",
        "file_path": "Hackathon final Samples/Malayalam_AI_GENERATED.mp3",
        "expected_classification": "AI_GENERATED"
    },
    {
        "language": "Tamil",
        "file_path": "Hackathon final Samples/TAMIL_VOICE__HUMAN.mp3",
        "expected_classification": "HUMAN"
    },
    {
        "language": "Telugu",
        "file_path": "Hackathon final Samples/Telugu_Voice_AI_GENERATED.mp3",
        "expected_classification": "AI_GENERATED"
    }
]


def test_api(endpoint_url=ENDPOINT_URL, api_key=API_KEY, test_files=TEST_FILES):
    total_files = len(test_files)
    score_per_file = 100 / total_files
    total_score = 0
    results = []

    print(f"\n{'='*60}")
    print(f"🚀 Starting Evaluation Against Hackathon Samples")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Total Test Files: {total_files}")
    print(f"Score per File: {score_per_file:.2f}")
    print(f"{'='*60}\n")

    for idx, file_data in enumerate(test_files):
        language = file_data["language"]
        file_path = file_data["file_path"]
        expected = file_data["expected_classification"]

        print(f"📝 Test {idx + 1}/{total_files}: {file_path}")
        print(f"   Expected: {expected} | Language: {language}")

        # Read and encode audio
        try:
            with open(file_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"   ❌ Failed to read file: {e}\n")
            results.append({"file": file_path, "status": "file_error", "score": 0})
            continue

        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }
        body = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }

        # Send request
        start = time.time()
        try:
            response = requests.post(endpoint_url, headers=headers, json=body, timeout=60)
            elapsed = time.time() - start
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout (>60s)\n")
            results.append({"file": file_path, "status": "timeout", "score": 0})
            continue
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Connection Error\n")
            results.append({"file": file_path, "status": "connection_error", "score": 0})
            continue

        print(f"   HTTP {response.status_code} | {elapsed:.2f}s")

        if response.status_code != 200:
            print(f"   ❌ Non-200 status: {response.text[:200]}\n")
            results.append({"file": file_path, "status": "http_error", "code": response.status_code, "score": 0})
            continue

        try:
            data = response.json()
        except:
            print(f"   ❌ Invalid JSON response\n")
            results.append({"file": file_path, "status": "json_error", "score": 0})
            continue

        # Validate response
        status = data.get("status", "")
        classification = data.get("classification", "")
        confidence = data.get("confidenceScore", None)

        if status != "success":
            print(f"   ❌ Status: {status}\n")
            results.append({"file": file_path, "status": "bad_status", "score": 0})
            continue

        if classification not in ["HUMAN", "AI_GENERATED"]:
            print(f"   ❌ Invalid classification: {classification}\n")
            results.append({"file": file_path, "status": "bad_classification", "score": 0})
            continue

        if confidence is None or not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            print(f"   ❌ Invalid confidence: {confidence}\n")
            results.append({"file": file_path, "status": "bad_confidence", "score": 0})
            continue

        # Score calculation
        file_score = 0
        if classification == expected:
            if confidence >= 0.8:
                file_score = score_per_file * 1.0
                tier = "100%"
            elif confidence >= 0.6:
                file_score = score_per_file * 0.75
                tier = "75%"
            elif confidence >= 0.4:
                file_score = score_per_file * 0.50
                tier = "50%"
            else:
                file_score = score_per_file * 0.25
                tier = "25%"
            total_score += file_score
            print(f"   ✅ {classification} (Correct!) | Confidence: {confidence:.2f} → {tier}")
            print(f"   🎯 Score: {file_score:.2f}/{score_per_file:.2f}\n")
        else:
            print(f"   ❌ {classification} (Expected: {expected}) | Confidence: {confidence:.2f}")
            print(f"   🎯 Score: 0/{score_per_file:.2f}\n")

        results.append({
            "file": file_path,
            "expected": expected,
            "actual": classification,
            "confidence": confidence,
            "correct": classification == expected,
            "score": round(file_score, 2),
            "time": round(elapsed, 2)
        })

    # Summary
    final_score = round(total_score)
    correct = sum(1 for r in results if r.get("correct", False))
    wrong = sum(1 for r in results if "correct" in r and not r["correct"])
    errors = sum(1 for r in results if "correct" not in r)

    print(f"{'='*60}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Correct: {correct}/{total_files}")
    print(f"❌ Wrong: {wrong}/{total_files}")
    print(f"⚠️  Errors: {errors}/{total_files}")
    print(f"🏆 Final Score: {final_score}/100")
    print(f"{'='*60}\n")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({"finalScore": final_score, "results": results}, f, indent=2)
    print(f"💾 Results saved to evaluation_results.json\n")

    return final_score


# Also test locally
def test_local(port=8000):
    print("\n🏠 Testing LOCAL endpoint...")
    return test_api(f"http://localhost:{port}/", API_KEY, TEST_FILES)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        test_local(port=int(sys.argv[2]) if len(sys.argv) > 2 else 7860)
    else:
        test_api()
