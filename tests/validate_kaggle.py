#!/usr/bin/env python3
"""
Validation Script for Kaggle Dataset
Tests the API with the downloaded labeled dataset.
"""
import os
import sys
import base64
import json
import random
from pathlib import Path
from typing import List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.audio_processor import audio_processor
from app.voice_detector import VoiceDetector

# Dataset path
DATASET_DIR = Path("/Users/garrry/.cache/kagglehub/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/versions/1")

def get_files(directory: Path, limit: int = 3) -> List[Path]:
    """Get audio files from directory recursively."""
    files = list(directory.rglob("*.wav")) + list(directory.rglob("*.mp3"))
    if not files:
        return []
    # If total files small, take all, otherwise simple random sample
    if len(files) <= limit:
        return files
    return random.sample(files, limit)

def validate_dataset():
    """Run validation on Kaggle dataset."""
    detector = VoiceDetector(use_ml=True)
    
    # Setup paths
    real_dir = DATASET_DIR / "Real"
    fake_dir = DATASET_DIR / "Fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        print(f"❌ Dataset directories not found at {DATASET_DIR}")
        return

    # Get samples
    real_files = get_files(real_dir)
    fake_files = get_files(fake_dir)
    
    print("=" * 70)
    print("KAGGLE DATASET VALIDATION (Ensemble Model)")
    print("=" * 70)
    print(f"Found {len(real_files)} Real samples")
    print(f"Found {len(fake_files)} Fake samples")
    print("-" * 70)
    
    results = {
        "true_positive": 0,  # AI correctly identified as AI
        "true_negative": 0,  # Human correctly identified as Human
        "false_positive": 0, # Human incorrectly identified as AI
        "false_negative": 0, # AI incorrectly identified as Human
        "total": 0,
        "details": []
    }
    
    # Test Real (Human) samples
    print("\n🎤 Testing REAL (Human) samples...")
    for filepath in real_files:
        _test_file(detector, filepath, "HUMAN", results)
        
    # Test Fake (AI) samples
    print("\n🤖 Testing FAKE (AI) samples...")
    for filepath in fake_files:
        _test_file(detector, filepath, "AI_GENERATED", results)
    
    # Statistics
    total = results["total"]
    if total == 0:
        print("No samples processed.")
        return

    accuracy = (results["true_positive"] + results["true_negative"]) / total
    
    # Precision (AI) = TP / (TP + FP)
    ai_precision = 0
    if (results["true_positive"] + results["false_positive"]) > 0:
        ai_precision = results["true_positive"] / (results["true_positive"] + results["false_positive"])
        
    # Recall (AI) = TP / (TP + FN)
    ai_recall = 0
    if (results["true_positive"] + results["false_negative"]) > 0:
        ai_recall = results["true_positive"] / (results["true_positive"] + results["false_negative"])
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total Samples: {total}")
    print(f"Accuracy:      {accuracy:.2%} ({results['true_positive'] + results['true_negative']}/{total})")
    print(f"AI Precision:  {ai_precision:.2%} (Reliability when saying 'AI')")
    print(f"AI Recall:     {ai_recall:.2%} (Ability to catch AI)")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f"              Predicted AI    Predicted Human")
    print(f"Actual AI     {results['true_positive']:<15} {results['false_negative']}")
    print(f"Actual Human  {results['false_positive']:<15} {results['true_negative']}")
    print("=" * 70)

def _test_file(detector, filepath, expected_label, results):
    try:
        with open(filepath, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        features, audio, sr = audio_processor.process_audio_with_samples(audio_b64)
        result = detector.detect(features, audio=audio, sr=sr)
        
        predicted = result['classification']
        conf = result['confidenceScore']
        
        results["total"] += 1
        
        is_correct = predicted == expected_label
        symbol = "✅" if is_correct else "❌"
        
        print(f"{symbol} {filepath.name}: {predicted} ({conf:.2f})")
        
        if expected_label == "AI_GENERATED":
            if predicted == "AI_GENERATED":
                results["true_positive"] += 1
            else:
                results["false_negative"] += 1
        else: # expected HUMAN
            if predicted == "HUMAN":
                results["true_negative"] += 1
            else:
                results["false_positive"] += 1
                
    except Exception as e:
        print(f"⚠️ Error processing {filepath.name}: {e}")

if __name__ == "__main__":
    validate_dataset()
