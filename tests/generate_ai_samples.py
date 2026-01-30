#!/usr/bin/env python3
"""
Generate AI voice samples using Google Text-to-Speech (gTTS)
These samples can be used to validate the detection accuracy.

Usage: 
    pip install gTTS
    python tests/generate_ai_samples.py
"""
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    print("Installing gTTS...")
    import subprocess
    subprocess.run(["pip", "install", "gTTS", "-q"])
    from gtts import gTTS

SAMPLES_DIR = Path("/Users/garrry/Downloads/Hackathon/test_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

# Sample texts in each language
SAMPLES = {
    "English": {
        "lang_code": "en",
        "texts": [
            "Hello, my name is John and I work as a software engineer in Bangalore. I love building artificial intelligence systems.",
            "Welcome to the hackathon. Today we will be learning about voice detection and machine learning algorithms.",
            "The weather today is beautiful with clear skies and a gentle breeze. It's perfect for outdoor activities.",
        ]
    },
    "Hindi": {
        "lang_code": "hi", 
        "texts": [
            "नमस्ते, मेरा नाम राहुल है और मैं बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता हूं।",
            "आज का मौसम बहुत अच्छा है और आसमान साफ है।",
            "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आजकल बहुत लोकप्रिय तकनीकें हैं।",
        ]
    },
    "Tamil": {
        "lang_code": "ta",
        "texts": [
            "வணக்கம், என் பெயர் குமார், நான் பெங்களூரில் மென்பொருள் பொறியாளராக பணிபுரிகிறேன்.",
            "இன்றைய வானிலை மிகவும் அழகாக இருக்கிறது.",
            "செயற்கை நுண்ணறிவு மற்றும் இயந்திர கற்றல் இன்று மிகவும் பிரபலமான தொழில்நுட்பங்கள்.",
        ]
    },
    "Telugu": {
        "lang_code": "te",
        "texts": [
            "నమస్కారం, నా పేరు రాజు, నేను బెంగళూరులో సాఫ్ట్‌వేర్ ఇంజనీర్‌గా పని చేస్తున్నాను.",
            "ఈ రోజు వాతావరణం చాలా అందంగా ఉంది.",
            "కృత్రిమ మేధస్సు మరియు మెషిన్ లెర్నింగ్ నేడు చాలా ప్రసిద్ధ టెక్నాలజీలు.",
        ]
    },
    "Malayalam": {
        "lang_code": "ml",
        "texts": [
            "നമസ്കാരം, എന്റെ പേര് അർജുൻ, ഞാൻ ബാംഗ്ലൂരിൽ സോഫ്റ്റ്‌വെയർ എഞ്ചിനീയറായി ജോലി ചെയ്യുന്നു.",
            "ഇന്നത്തെ കാലാവസ്ഥ വളരെ മനോഹരമാണ്.",
            "ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസും മെഷീൻ ലേണിംഗും ഇന്ന് വളരെ ജനപ്രിയമായ സാങ്കേതികവിദ്യകളാണ്.",
        ]
    }
}


def generate_samples():
    """Generate AI voice samples for all languages."""
    print("=" * 60)
    print("GENERATING AI VOICE SAMPLES (gTTS)")
    print("=" * 60)
    
    generated_files = []
    
    for language, data in SAMPLES.items():
        lang_code = data["lang_code"]
        texts = data["texts"]
        
        print(f"\n📝 {language} ({lang_code}):")
        
        for i, text in enumerate(texts, 1):
            filename = f"{language.lower()}_ai_{i}.mp3"
            filepath = SAMPLES_DIR / filename
            
            try:
                tts = gTTS(text=text, lang=lang_code, slow=False)
                tts.save(str(filepath))
                print(f"   ✅ Created: {filename}")
                generated_files.append(filename)
            except Exception as e:
                print(f"   ❌ Failed: {filename} - {e}")
    
    print(f"\n{'=' * 60}")
    print(f"✅ Generated {len(generated_files)} AI samples")
    print(f"📁 Location: {SAMPLES_DIR}")
    print(f"\n⚠️  Note: These are Google TTS samples (AI-generated)")
    print(f"   Expected label: AI_GENERATED")
    print(f"\n💡 Next: Add human voice samples to the same folder")
    print(f"   Download from: OpenSLR, CommonVoice, or record yourself")
    
    return generated_files


if __name__ == "__main__":
    generate_samples()
