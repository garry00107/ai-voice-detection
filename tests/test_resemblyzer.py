import os
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# Initialize encoder (downloads model on first run)
print("Initializing VoiceEncoder...")
encoder = VoiceEncoder()
print("Encoder loaded.")

def get_stability_score(filepath):
    try:
        # Load and preprocess
        fpath = Path(filepath)
        wav = preprocess_wav(fpath)
        
        # Split into 3 segments
        n = len(wav)
        if n < 16000: return None # Too short
        
        chunk_size = n // 3
        chunks = [wav[i:i+chunk_size] for i in range(0, n - (n%3), chunk_size)]
        if len(chunks) < 3: return None
        
        # Compute embeddings
        embeds = [encoder.embed_utterance(chunk) for chunk in chunks]
        
        # Calculate pair-wise similarity
        sims = []
        sims.append(np.dot(embeds[0], embeds[1]))
        sims.append(np.dot(embeds[1], embeds[2]))
        sims.append(np.dot(embeds[0], embeds[2]))
        
        return np.mean(sims)
    except Exception as e:
        print(f"Error {filepath}: {e}")
        return None

print("\n--- Testing Human Samples ---")
human_dir = "human-nonhuman/human"
human_scores = []
for f in sorted(os.listdir(human_dir))[:5]:
    if not f.endswith('.mp3'): continue
    path = os.path.join(human_dir, f)
    score = get_stability_score(path)
    if score is not None:
        human_scores.append(score)
        print(f"{f}: {score:.4f}")

print(f"Average Human Stability: {np.mean(human_scores):.4f}")

print("\n--- Testing AI Samples ---")
ai_dir = "human-nonhuman/nonhuman"
ai_scores = []
for f in sorted(os.listdir(ai_dir))[:5]:
    if not f.endswith('.mp3'): continue
    path = os.path.join(ai_dir, f)
    score = get_stability_score(path)
    if score is not None:
        ai_scores.append(score)
        print(f"{f}: {score:.4f}")

print(f"Average AI Stability: {np.mean(ai_scores):.4f}")
