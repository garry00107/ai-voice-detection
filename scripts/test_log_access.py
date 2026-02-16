
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
repo_id = "gaurav00107/ai-voice-detection"

print(f"Fetching logs for {repo_id}...")

try:
    api = HfApi(token=hf_token)
    
    # Try to get logs - the method might be get_space_logs or similar
    # If not directly available in this version, we might have to use internal API or just runtime info
    # Let's try listing files first to see if there's a log file exposed (unlikely)
    print("Files in repo:")
    files = api.list_repo_files(repo_id=repo_id, repo_type="space")
    print(files[:10])
    
    # Check runtime
    runtime = api.get_space_runtime(repo_id=repo_id)
    print(f"Runtime status: {runtime.stage}")
    
    # There isn't a direct 'get_logs' in simple HfApi usually, it's often via the UI or websocket
    # But there is a way to get the logs via the API https://huggingface.co/api/spaces/{repo_id}/logs
    import requests
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get(f"https://huggingface.co/api/spaces/{repo_id}/runtime", headers=headers)
    print("Runtime API response:", response.json())
    
except Exception as e:
    print(f"Error: {e}")
