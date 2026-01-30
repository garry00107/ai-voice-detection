#!/usr/bin/env python3
"""
Download human voice samples from OpenSLR and other free sources.
These are real human recordings for validating AI detection accuracy.
"""
import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import random
import shutil

SAMPLES_DIR = Path("/Users/garrry/Downloads/Hackathon/test_samples")
TEMP_DIR = Path("/Users/garrry/Downloads/Hackathon/temp_downloads")

SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# OpenSLR datasets for Indian languages
OPENSLR_DATASETS = {
    "Tamil": {
        "url": "https://www.openslr.org/resources/65/ta_in_female.zip",
        "name": "SLR65 Tamil Female",
    },
    "Telugu": {
        "url": "https://www.openslr.org/resources/66/te_in_female.zip",
        "name": "SLR66 Telugu Female",
    },
    "Malayalam": {
        "url": "https://www.openslr.org/resources/63/ml_in_female.zip",
        "name": "SLR63 Malayalam Female",
    },
}


def download_file(url, dest_path):
    """Download a file with progress."""
    print(f"   Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False


def extract_samples_from_zip(zip_path, language, num_samples=2):
    """Extract a few audio samples from a zip file."""
    extracted = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find audio files  
            audio_files = [f for f in zf.namelist() if f.endswith('.wav') or f.endswith('.mp3')]
            
            if not audio_files:
                print(f"   ⚠️  No audio files found in archive")
                return extracted
            
            # Select random samples
            selected = random.sample(audio_files, min(num_samples, len(audio_files)))
            
            for i, audio_file in enumerate(selected, 1):
                # Extract to temp
                zf.extract(audio_file, TEMP_DIR)
                src = TEMP_DIR / audio_file
                
                # Determine extension
                ext = Path(audio_file).suffix
                dest_name = f"{language.lower()}_human_{i}{ext}"
                dest = SAMPLES_DIR / dest_name
                
                # Move to samples dir
                shutil.move(str(src), str(dest))
                extracted.append(dest_name)
                print(f"   ✅ Extracted: {dest_name}")
                
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
    
    return extracted


def download_librivox_english():
    """Download a sample from LibriVox (public domain English audiobooks)."""
    print("\n📥 Downloading English human samples (LibriVox)...")
    
    # LibriVox sample URLs (public domain)
    urls = [
        "https://ia800500.us.archive.org/23/items/gettysburg_address_1223/gettysburg_address_1223_librivox.mp3",
    ]
    
    extracted = []
    for i, url in enumerate(urls, 1):
        dest = SAMPLES_DIR / f"english_human_{i}.mp3"
        if download_file(url, dest):
            extracted.append(f"english_human_{i}.mp3")
            print(f"   ✅ Downloaded: english_human_{i}.mp3")
    
    return extracted


def create_recording_instructions():
    """Create instructions for recording your own samples."""
    instructions = SAMPLES_DIR / "RECORDING_INSTRUCTIONS.txt"
    
    with open(instructions, 'w') as f:
        f.write("""
=====================================================
HOW TO RECORD YOUR OWN HUMAN VOICE SAMPLES
=====================================================

If downloads fail or you want more samples, you can record yourself!

QUICK RECORDING (Recommended):
1. Use Voice Memos on your phone
2. Speak clearly for 5-10 seconds in each language
3. Transfer to your computer
4. Save as: {language}_human_X.mp3

SAMPLE SCRIPTS:

English:
"Hello, my name is [Your Name]. I am testing the voice detection 
system for a hackathon project. This is a sample recording."

Hindi:
"नमस्ते, मेरा नाम [आपका नाम] है। मैं हैकाथॉन प्रोजेक्ट के लिए 
वॉइस डिटेक्शन सिस्टम का परीक्षण कर रहा हूं।"

Tamil:
"வணக்கம், என் பெயர் [உங்கள் பெயர்]. நான் ஒரு ஹேக்கத்தான் 
திட்டத்திற்காக குரல் கண்டறிதல் அமைப்பை சோதிக்கிறேன்."

Telugu:
"నమస్కారం, నా పేరు [మీ పేరు]. నేను హ్యాకథాన్ ప్రాజెక్ట్ కోసం 
వాయిస్ డిటెక్షన్ సిస్టమ్‌ను టెస్ట్ చేస్తున్నాను."

Malayalam:
"നമസ്കാരം, എന്റെ പേര് [നിങ്ങളുടെ പേര്]. ഞാൻ ഒരു ഹാക്കത്തോൺ 
പ്രോജക്റ്റിനായി വോയ്‌സ് ഡിറ്റക്ഷൻ സിസ്റ്റം ടെസ്റ്റ് ചെയ്യുകയാണ്."

TIPS:
- Speak naturally, don't try to sound robotic
- Use a quiet environment
- Hold phone/mic about 6 inches from mouth
- Each recording should be 5-15 seconds

""")
    print(f"\n📝 Recording instructions saved to: {instructions}")


def main():
    print("=" * 60)
    print("DOWNLOADING HUMAN VOICE SAMPLES")
    print("=" * 60)
    
    all_downloaded = []
    
    # Try to download from OpenSLR (large files, may take time)
    print("\n⚠️  Note: OpenSLR datasets are large (100MB+)")
    print("   Downloading just sample files instead...\n")
    
    # Download English sample
    english = download_librivox_english()
    all_downloaded.extend(english)
    
    # For Indian languages, since full datasets are huge,
    # let's provide alternative instructions
    create_recording_instructions()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    if all_downloaded:
        print(f"\n✅ Downloaded {len(all_downloaded)} human samples:")
        for f in all_downloaded:
            print(f"   - {f}")
    
    print("""
💡 For Indian language human samples:
   
   Option 1: Record yourself (see RECORDING_INSTRUCTIONS.txt)
   
   Option 2: Download from Mozilla Common Voice:
   - Hindi: https://commonvoice.mozilla.org/hi/datasets
   - Tamil: https://commonvoice.mozilla.org/ta/datasets
   
   Option 3: Download from OpenSLR manually:
   - Tamil: https://openslr.org/65/
   - Telugu: https://openslr.org/66/
   - Malayalam: https://openslr.org/63/
   
   After downloading, extract 1-2 audio files and save as:
   - hindi_human_1.mp3
   - tamil_human_1.mp3
   - telugu_human_1.mp3
   - malayalam_human_1.mp3
""")
    
    # Cleanup
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    return all_downloaded


if __name__ == "__main__":
    main()
