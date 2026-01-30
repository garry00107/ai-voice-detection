#!/usr/bin/env python3
"""
Validation Script for AI Voice Detection API
Tests the API with labeled samples and calculates accuracy.
"""
import os
import sys
import base64
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.audio_processor import audio_processor
from app.voice_detector import VoiceDetector


# === LABELED TEST SAMPLES ===
# Add your test samples here with their known labels
# Format: (filename, language, expected_label, source_description)
#
# Place your audio files in /Users/garrry/Downloads/Hackathon/test_samples/
#
TEST_SAMPLES = [
    # AI-Generated samples (from TTS services)
    ("english_ai_1.mp3", "English", "AI_GENERATED", "Google TTS"),
    ("english_ai_2.mp3", "English", "AI_GENERATED", "ElevenLabs"),
    ("hindi_ai_1.mp3", "Hindi", "AI_GENERATED", "Google TTS Hindi"),
    ("tamil_ai_1.mp3", "Tamil", "AI_GENERATED", "Google TTS Tamil"),
    ("telugu_ai_1.mp3", "Telugu", "AI_GENERATED", "Google TTS Telugu"),
    ("malayalam_ai_1.mp3", "Malayalam", "AI_GENERATED", "Google TTS Malayalam"),
    
    # Human samples (from recordings or datasets)
    ("english_human_1.mp3", "English", "HUMAN", "LibriSpeech"),
    ("english_human_2.mp3", "English", "HUMAN", "Personal recording"),
    ("hindi_human_1.mp3", "Hindi", "HUMAN", "CommonVoice Hindi"),
    ("tamil_human_1.mp3", "Tamil", "HUMAN", "OpenSLR Tamil"),
    ("telugu_human_1.mp3", "Telugu", "HUMAN", "OpenSLR Telugu"),
    ("malayalam_human_1.mp3", "Malayalam", "HUMAN", "OpenSLR Malayalam"),
    
    # The hackathon sample
    ("sample voice 1.mp3", "English", "AI_GENERATED", "Hackathon Sample"),
]

SAMPLES_DIR = Path("/Users/garrry/Downloads/Hackathon/test_samples")


def validate_samples():
    """Run validation on all available test samples."""
    detector = VoiceDetector()
    
    results = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
        "by_language": {},
        "by_type": {"AI_GENERATED": {"correct": 0, "total": 0}, "HUMAN": {"correct": 0, "total": 0}},
        "details": []
    }
    
    # Initialize language stats
    for lang in ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]:
        results["by_language"][lang] = {"correct": 0, "total": 0}
    
    print("=" * 70)
    print("AI VOICE DETECTION VALIDATION")
    print("=" * 70)
    
    for filename, language, expected_label, source in TEST_SAMPLES:
        # Check in samples dir first, then in main Hackathon folder
        filepath = SAMPLES_DIR / filename
        if not filepath.exists():
            filepath = Path("/Users/garrry/Downloads/Hackathon") / filename
        
        if not filepath.exists():
            results["missing"] += 1
            print(f"\n⚠️  MISSING: {filename}")
            continue
        
        # Load and process audio
        print(f"\n📁 Testing: {filename}")
        print(f"   Language: {language} | Expected: {expected_label} | Source: {source}")
        
        try:
            with open(filepath, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            features = audio_processor.process_audio(audio_b64)
            result = detector.detect(features)
            predicted_label = result['classification']
            confidence = result['confidenceScore']
            
            is_correct = predicted_label == expected_label
            results["total"] += 1
            results["by_language"][language]["total"] += 1
            results["by_type"][expected_label]["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                results["by_language"][language]["correct"] += 1
                results["by_type"][expected_label]["correct"] += 1
                print(f"   ✅ CORRECT: {predicted_label} (confidence: {confidence})")
            else:
                results["incorrect"] += 1
                print(f"   ❌ WRONG: Predicted {predicted_label}, Expected {expected_label} (confidence: {confidence})")
            
            results["details"].append({
                "file": filename,
                "language": language,
                "expected": expected_label,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
                "explanation": result['explanation']
            })
            
        except Exception as e:
            print(f"   ⚠️  ERROR: {str(e)}")
            results["missing"] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if results["total"] > 0:
        accuracy = (results["correct"] / results["total"]) * 100
        print(f"\n🎯 OVERALL ACCURACY: {accuracy:.1f}% ({results['correct']}/{results['total']})")
        
        print("\n📊 By Language:")
        for lang, stats in results["by_language"].items():
            if stats["total"] > 0:
                lang_acc = (stats["correct"] / stats["total"]) * 100
                print(f"   {lang}: {lang_acc:.1f}% ({stats['correct']}/{stats['total']})")
        
        print("\n📊 By Type:")
        for label, stats in results["by_type"].items():
            if stats["total"] > 0:
                type_acc = (stats["correct"] / stats["total"]) * 100
                print(f"   {label}: {type_acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n⚠️  Missing samples: {results['missing']}")
    
    # Save results to JSON
    results_file = Path("/Users/garrry/Downloads/Hackathon/validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
    
    return results


def print_sample_guide():
    """Print guide for obtaining test samples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           HOW TO GET TEST SAMPLES FOR VALIDATION                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CREATE THIS FOLDER:                                                 ║
║  /Users/garrry/Downloads/Hackathon/test_samples/                     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  AI-GENERATED SAMPLES (TTS Services - FREE):                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Google Cloud TTS (Free Tier):                                   ║
║     https://cloud.google.com/text-to-speech                          ║
║     - Supports: English, Hindi, Tamil, Telugu, Malayalam             ║
║     - Use the demo page to generate samples                          ║
║                                                                      ║
║  2. Sarvam AI Bulbul (Free Credits):                                ║
║     https://sarvam.ai/                                               ║
║     - Supports 11 Indian languages                                   ║
║                                                                      ║
║  3. ElevenLabs (Free Tier):                                         ║
║     https://elevenlabs.io/                                           ║
║     - Very realistic English AI voices                               ║
║                                                                      ║
║  4. Microsoft Azure TTS (Free Tier):                                ║
║     https://azure.microsoft.com/en-us/services/cognitive-services/   ║
║     text-to-speech/                                                  ║
║     - Supports Indian languages                                      ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  HUMAN SAMPLES (Free Datasets):                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Mozilla Common Voice:                                            ║
║     https://commonvoice.mozilla.org/en/datasets                      ║
║     - Hindi, Tamil available                                         ║
║                                                                      ║
║  2. OpenSLR:                                                         ║
║     https://openslr.org/resources.php                                ║
║     - SLR65: Tamil   - SLR66: Telugu   - SLR67: Malayalam           ║
║                                                                      ║
║  3. LibriSpeech (English):                                          ║
║     https://www.openslr.org/12                                       ║
║                                                                      ║
║  4. Your own recordings:                                             ║
║     - Record yourself or friends speaking                            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SAMPLE TEXT TO USE FOR TTS:                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  English: "Hello, my name is John and I work as a software          ║
║           engineer in Bangalore."                                    ║
║                                                                      ║
║  Hindi:   "नमस्ते, मेरा नाम राहुल है और मैं बैंगलोर में              ║
║           सॉफ्टवेयर इंजीनियर के रूप में काम करता हूं।"               ║
║                                                                      ║
║  Tamil:   "வணக்கம், என் பெயர் குமார், நான் பெங்களூரில்                ║
║           மென்பொருள் பொறியாளராக பணிபுரிகிறேன்."                      ║
║                                                                      ║
║  Telugu:  "నమస్కారం, నా పేరు రాజు, నేను బెంగళూరులో                   ║
║           సాఫ్ట్‌వేర్ ఇంజనీర్‌గా పని చేస్తున్నాను."                   ║
║                                                                      ║
║  Malayalam: "നമസ്കാരം, എന്റെ പേര് അർജുൻ, ഞാൻ ബാംഗ്ലൂരിൽ              ║
║              സോഫ്റ്റ്‌വെയർ എഞ്ചിനീയറായി ജോലി ചെയ്യുന്നു."            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate AI Voice Detection API")
    parser.add_argument("--guide", action="store_true", help="Show guide for getting samples")
    args = parser.parse_args()
    
    if args.guide:
        print_sample_guide()
    else:
        # Create samples directory if it doesn't exist
        SAMPLES_DIR.mkdir(exist_ok=True)
        validate_samples()
