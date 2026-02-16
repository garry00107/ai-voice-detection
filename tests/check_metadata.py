import os
import subprocess
import json

def get_metadata(filepath):
    """Refers to ffprobe to get metadata"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data.get('format', {}).get('tags', {})
    except Exception as e:
        return {}

print("Scanning AI Samples for Metadata clues...")
ai_dir = "human-nonhuman/nonhuman"
suspicious_tags = ['lavfi', 'encoder', 'comment', 'artist', 'title']

count = 0
found_clues = 0

for f in sorted(os.listdir(ai_dir))[:20]:
    if not f.endswith(('.mp3', '.wav')): continue
    path = os.path.join(ai_dir, f)
    tags = get_metadata(path)
    
    # Check for keywords
    clue_found = False
    for k, v in tags.items():
        v_lower = str(v).lower()
        if 'lavfi' in v_lower: clue_found = True
        if 'lame' in v_lower: clue_found = True # Common in mp3 encoding output
        if 'elevenlabs' in v_lower: clue_found = True
        
    if clue_found or len(tags) > 0:
        print(f"\nExample: {f}")
        print(f"Tags: {tags}")
        found_clues += 1
    
    count += 1

print(f"\nFound significant metadata in {found_clues}/{count} samples.")
