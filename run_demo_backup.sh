#!/bin/bash
# Backup Demo Script for Grand Finale
# Run this if HuggingFace Space is down or network is unavailable

echo "🚀 Starting AI Voice Detection - Local Backup Demo"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run 'python -m venv venv' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Kill any existing Gradio instances
pkill -f gradio_app 2>/dev/null || true

# Start the Gradio demo
echo "🎨 Starting Gradio Demo on http://localhost:7860"
echo ""
echo "📌 Share this with judges: http://YOUR_IP:7860"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python gradio_app.py
