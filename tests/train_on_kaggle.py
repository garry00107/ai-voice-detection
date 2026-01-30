#!/usr/bin/env python3
"""
Training Script
Trains a Random Forest classifier on the Kaggle dataset (Few-Shot Learning).
Uses Wav2Vec2 embeddings as features.
"""
import os
import sys
import base64
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.audio_processor import audio_processor
from app.ml_detector import get_ml_detector

# Dataset path
DATASET_DIR = Path("/Users/garrry/.cache/kagglehub/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/versions/1")
MODEL_OUTPUT_PATH = Path("app/trained_model.joblib")

def get_files(directory: Path) -> list[Path]:
    """Get all audio files recursively."""
    return list(directory.rglob("*.wav")) + list(directory.rglob("*.mp3"))

def train_model():
    print("=" * 60)
    print("TRAINING MODE: Fine-tuning on Kaggle Dataset")
    print("=" * 60)
    
    # Initialize ML detector
    ml_detector = get_ml_detector()
    ml_detector.load_model()
    
    # 1. Load Data
    real_dir = DATASET_DIR / "Real"
    fake_dir = DATASET_DIR / "Fake"
    
    real_files = get_files(real_dir)
    fake_files = get_files(fake_dir)
    
    print(f"Found {len(real_files)} Real samples")
    print(f"Found {len(fake_files)} Fake samples")
    
    X = [] # Features
    y = [] # Labels (0=Human, 1=AI)
    files_processed = []
    
    # 2. Extract Embeddings
    print("\nExtracting features (Wav2Vec2 embeddings)...")
    
    # Process Real (Human) -> Label 0
    for i, filepath in enumerate(real_files):
        print(f"\rProcessing Real [{i+1}/{len(real_files)}]", end="")
        try:
            with open(filepath, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            _, audio, sr = audio_processor.process_audio_with_samples(audio_b64)
            
            # Resample needed? Wav2Vec2 handles this inside extract_wav2vec_features usually via processor
            # But let's ensure we pass 16k if needed. Our ML detector handles resampling if passed raw audio? 
            # Actually ml_detector.extract_wav2vec_features expects 16k.
            import librosa
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            embeddings = ml_detector.extract_wav2vec_features(audio, 16000)
            
            if embeddings is not None:
                # We can use the mean-pooled embeddings (768 dim) directly
                X.append(embeddings)
                y.append(0) # HUMAN
                files_processed.append(filepath.name)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            
    print("\n")
    
    # Process Fake (AI) -> Label 1
    for i, filepath in enumerate(fake_files):
        print(f"\rProcessing Fake [{i+1}/{len(fake_files)}]", end="")
        try:
            with open(filepath, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            _, audio, sr = audio_processor.process_audio_with_samples(audio_b64)
            
            import librosa
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            embeddings = ml_detector.extract_wav2vec_features(audio, 16000)
            
            if embeddings is not None:
                X.append(embeddings)
                y.append(1) # AI
                files_processed.append(filepath.name)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")

    print(f"\n\nTotal samples processed: {len(X)}")
    
    if len(X) == 0:
        print("No data processed. Exiting.")
        return

    X = np.array(X)
    y = np.array(y)
    
    # 3. Train Classifier
    # Using Random Forest as it works well with high-dim features and small datasets
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train heavily on all data (since we have small dataset, let's use all for reliable model)
    # But let's do a quick CV score first
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-Validation Accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
    
    # Train on full dataset
    clf.fit(X, y)
    
    # 4. Save Model
    print(f"\nSaving model to {MODEL_OUTPUT_PATH}...")
    model_data = {
        'model': clf,
        'type': 'rf_wav2vec2',
        'version': '1.0'
    }
    joblib.dump(model_data, MODEL_OUTPUT_PATH)
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
