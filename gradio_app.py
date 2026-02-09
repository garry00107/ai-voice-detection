"""
AI Voice Detection - Gradio Demo UI
Interactive demo for the India AI Impact Buildathon Grand Finale
Enhanced UI with custom dark theme and better visualization
"""
import gradio as gr
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import tempfile
import os

# Import our detection modules
from app.audio_processor import audio_processor
from app.voice_detector import voice_detector


# Custom CSS for polished dark theme
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white !important;
    font-size: 2.5em !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1em !important;
}

/* Result box animations */
.result-box {
    min-height: 200px;
    transition: all 0.3s ease;
}

.result-ai {
    animation: pulseRed 2s infinite;
}

.result-human {
    animation: pulseGreen 2s infinite;
}

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.4); }
    50% { box-shadow: 0 0 40px rgba(255, 68, 68, 0.8); }
}

@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(68, 255, 68, 0.4); }
    50% { box-shadow: 0 0 40px rgba(68, 255, 68, 0.8); }
}

/* Button styling */
.primary-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.2em !important;
    padding: 15px 30px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
}

/* Card styling */
.feature-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}

/* Language badges */
.lang-badge {
    display: inline-block;
    padding: 5px 15px;
    margin: 3px;
    border-radius: 20px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 0.9em;
}

/* Confidence meter */
.confidence-meter {
    height: 20px;
    border-radius: 10px;
    background: #1a1a2e;
    overflow: hidden;
    margin: 10px 0;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}
"""


def generate_spectrogram(audio: np.ndarray, sr: int):
    """Generate mel-spectrogram with dark theme"""
    try:
        from PIL import Image
        
        # Use dark theme for spectrogram
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 4))
        
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram Analysis', fontsize=14, color='white', pad=10)
        
        # Style the axes
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                    facecolor='#1a1a2e', edgecolor='none')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    except Exception as e:
        print(f"Spectrogram error: {e}")
        return None


def create_result_html(classification: str, confidence: float, duration: float):
    """Create beautiful result display with animations"""
    if classification == "AI_GENERATED":
        emoji = "🤖"
        color = "#ff4444"
        gradient = "linear-gradient(135deg, #ff4444 0%, #cc0000 100%)"
        label = "AI-Generated Voice"
        icon = "⚠️"
        message = "This audio appears to be synthetically generated"
    else:
        emoji = "👤"
        color = "#44ff88"
        gradient = "linear-gradient(135deg, #44ff88 0%, #00cc66 100%)"
        label = "Human Voice"
        icon = "✅"
        message = "This audio appears to be authentic human speech"
    
    # Confidence bar color
    if confidence > 0.8:
        bar_color = "#44ff88" if classification == "HUMAN" else "#ff4444"
    elif confidence > 0.6:
        bar_color = "#ffaa44"
    else:
        bar_color = "#ffff44"
    
    return f"""
    <div style="
        text-align: center; 
        padding: 30px; 
        border-radius: 20px; 
        background: {gradient};
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        margin: 10px 0;
    ">
        <div style="font-size: 64px; margin-bottom: 10px;">{emoji}</div>
        <h2 style="color: white; margin: 10px 0; font-size: 28px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            {icon} {label}
        </h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 16px; margin-bottom: 20px;">
            {message}
        </p>
        
        <!-- Confidence Meter -->
        <div style="
            background: rgba(0,0,0,0.3); 
            border-radius: 15px; 
            padding: 15px;
            margin-top: 15px;
        ">
            <p style="color: white; margin: 0 0 10px 0; font-size: 14px;">Confidence Level</p>
            <div style="
                height: 25px; 
                background: rgba(0,0,0,0.4); 
                border-radius: 12px; 
                overflow: hidden;
                border: 2px solid rgba(255,255,255,0.2);
            ">
                <div style="
                    width: {confidence*100}%; 
                    height: 100%; 
                    background: linear-gradient(90deg, {bar_color}, white);
                    border-radius: 10px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <p style="color: white; font-size: 32px; font-weight: bold; margin: 10px 0 0 0;">
                {confidence*100:.1f}%
            </p>
        </div>
    </div>
    
    <div style="
        display: flex; 
        justify-content: center; 
        gap: 20px; 
        margin-top: 15px;
    ">
        <div style="
            background: rgba(255,255,255,0.1); 
            padding: 10px 20px; 
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        ">
            <span style="color: rgba(255,255,255,0.7);">Duration:</span>
            <span style="color: white; font-weight: bold;"> {duration:.2f}s</span>
        </div>
    </div>
    """


def detect_voice(audio_input, language: str):
    """Main detection function for Gradio interface"""
    if audio_input is None:
        return None, """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 48px;">🎤</div>
            <p>Upload an audio file or record your voice to analyze</p>
        </div>
        """, ""
    
    try:
        # Handle tuple input from Gradio (sample_rate, audio_array)
        if isinstance(audio_input, tuple):
            sample_rate, audio_array = audio_input
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            if sample_rate != 22050:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=22050)
                sample_rate = 22050
        else:
            audio_array, sample_rate = librosa.load(audio_input, sr=22050, mono=True)
        
        duration = len(audio_array) / sample_rate
        if duration < 0.5:
            return None, """
            <div style="text-align: center; padding: 30px; background: #442222; border-radius: 15px;">
                <div style="font-size: 48px;">⚠️</div>
                <p style="color: #ff8888;">Audio too short! Minimum 0.5 seconds required.</p>
            </div>
            """, ""
        
        # Convert to bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, sample_rate)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            os.unlink(tmp.name)
        
        # Extract features and detect
        features, audio_samples, sr = audio_processor.process_audio_with_samples(audio_base64)
        result = voice_detector.detect(
            features=features,
            audio=audio_samples,
            sr=sr,
            audio_bytes=audio_bytes
        )
        
        # Generate outputs
        spec_image = generate_spectrogram(audio_array, sample_rate)
        classification = result['classification']
        confidence = result['confidenceScore']
        explanation = result.get('explanation', '')
        
        result_html = create_result_html(classification, confidence, duration)
        
        # Model breakdown
        details = f"""
### 🔬 Technical Analysis

**Explanation**: {explanation}

---

### 📊 Model Scores
"""
        if 'model_scores' in result:
            for model, score in result['model_scores'].items():
                model_name = model.replace('_', ' ').title()
                details += f"- **{model_name}**: {score:.1%}\n"
        
        return spec_image, result_html, details
        
    except Exception as e:
        import traceback
        return None, f"""
        <div style="text-align: center; padding: 30px; background: #442222; border-radius: 15px;">
            <div style="font-size: 48px;">❌</div>
            <p style="color: #ff8888;">Error: {str(e)}</p>
            <pre style="color: #888; font-size: 12px; text-align: left; overflow: auto;">{traceback.format_exc()}</pre>
        </div>
        """, ""


# Create the enhanced Gradio interface
with gr.Blocks(
    title="AI Voice Detection | India AI Impact Buildathon",
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="blue", 
        neutral_hue="slate",
    ).set(
        body_background_fill="#0f0f1a",
        body_background_fill_dark="#0f0f1a",
        block_background_fill="#1a1a2e",
        block_background_fill_dark="#1a1a2e",
        border_color_primary="#333355",
        button_primary_background_fill="linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #7790ff 0%, #8860b5 100%)",
    ),
    css=CUSTOM_CSS
) as demo:
    
    # Header
    gr.HTML("""
    <div style="
        text-align: center;
        padding: 30px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    ">
        <h1 style="color: white; font-size: 2.8em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🛡️ AI Voice Detection System
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.2em; margin: 10px 0 0 0;">
            India AI Impact Buildathon 2026 | Grand Finale Demo
        </p>
        <div style="margin-top: 15px;">
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; margin: 0 5px;">🇮🇳 Tamil</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; margin: 0 5px;">🇮🇳 English</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; margin: 0 5px;">🇮🇳 Hindi</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; margin: 0 5px;">🇮🇳 Malayalam</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; margin: 0 5px;">🇮🇳 Telugu</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="
                background: rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 15px;
                padding: 15px;
                margin-bottom: 15px;
            ">
                <h3 style="color: #667eea; margin: 0 0 10px 0;">📤 Upload Audio</h3>
                <p style="color: #888; margin: 0; font-size: 0.9em;">
                    Upload an MP3/WAV file or record directly from your microphone
                </p>
            </div>
            """)
            
            audio_input = gr.Audio(
                label="",
                sources=["upload", "microphone"],
                type="numpy"
            )
            
            language = gr.Dropdown(
                choices=["English", "Tamil", "Hindi", "Malayalam", "Telugu"],
                value="English",
                label="🌍 Language"
            )
            
            detect_btn = gr.Button(
                "🔍 Analyze Voice",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"]
            )
            
            gr.HTML("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 20px;
                margin-top: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h4 style="color: #888; margin: 0 0 15px 0;">💡 Quick Tips</h4>
                <ul style="color: #666; margin: 0; padding-left: 20px; font-size: 0.9em;">
                    <li>Minimum 0.5 seconds of audio</li>
                    <li>Clear speech works best</li>
                    <li>Supports MP3, WAV, OGG formats</li>
                </ul>
            </div>
            """)
        
        # Right Column - Results
        with gr.Column(scale=2):
            result_html = gr.HTML(
                value="""
                <div style="text-align: center; padding: 60px; color: #555;">
                    <div style="font-size: 64px; margin-bottom: 20px;">🎤</div>
                    <p style="font-size: 1.2em;">Upload an audio file to analyze</p>
                    <p style="color: #444; font-size: 0.9em;">Our AI will detect if the voice is human or AI-generated</p>
                </div>
                """,
                elem_classes=["result-box"]
            )
            
            # Load placeholder image for spectrogram
            import os
            placeholder_path = os.path.join(os.path.dirname(__file__), "assets", "spectrogram_placeholder.png")
            placeholder_img = None
            if os.path.exists(placeholder_path):
                from PIL import Image
                placeholder_img = Image.open(placeholder_path)
            
            spectrogram = gr.Image(
                label="📊 Mel-Spectrogram Visualization",
                type="pil",
                value=placeholder_img
            )
            
            details_md = gr.Markdown()
    
    # How It Works Section
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    ">
        <h2 style="color: #667eea; text-align: center; margin-bottom: 25px;">🔬 How It Works</h2>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 15px;
                width: 280px;
                text-align: center;
            ">
                <div style="font-size: 40px; margin-bottom: 10px;">🤖</div>
                <h4 style="color: white; margin: 0 0 10px 0;">Wav2Vec2 Transformer</h4>
                <p style="color: #888; font-size: 0.85em; margin: 0;">Deep learning embeddings for detecting synthetic artifacts</p>
                <div style="color: #667eea; font-weight: bold; margin-top: 10px;">60% Weight</div>
            </div>
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 15px;
                width: 280px;
                text-align: center;
            ">
                <div style="font-size: 40px; margin-bottom: 10px;">📊</div>
                <h4 style="color: white; margin: 0 0 10px 0;">CNN on MFCC</h4>
                <p style="color: #888; font-size: 0.85em; margin: 0;">Neural network analyzing spectral patterns</p>
                <div style="color: #667eea; font-weight: bold; margin-top: 10px;">35% Weight</div>
            </div>
            <div style="
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 15px;
                width: 280px;
                text-align: center;
            ">
                <div style="font-size: 40px; margin-bottom: 10px;">📈</div>
                <h4 style="color: white; margin: 0 0 10px 0;">Heuristic Analysis</h4>
                <p style="color: #888; font-size: 0.85em; margin: 0;">Statistical analysis of pitch, rhythm & entropy</p>
                <div style="color: #667eea; font-weight: bold; margin-top: 10px;">30% Weight</div>
            </div>
        </div>
    """)
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; color: #555;">
        <p style="margin: 0;">Built with ❤️ for <strong>India AI Impact Buildathon 2026</strong></p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Team: Gaurav Sulsule | github.com/garry00107</p>
    </div>
    """)
    
    # Connect button
    detect_btn.click(
        fn=detect_voice,
        inputs=[audio_input, language],
        outputs=[spectrogram, result_html, details_md]
    )


# ============ BATCH PROCESSING APP ============
def process_batch(files, language):
    """Process multiple audio files"""
    if not files:
        return "<div style='text-align:center;padding:40px;color:#888;'>📁 Upload files to analyze</div>"
    
    results = []
    ai_count = 0
    human_count = 0
    
    for file in files:
        try:
            audio, sr = librosa.load(file, sr=22050, mono=True)
            duration = len(audio) / sr
            if duration < 0.5:
                results.append(f"⚠️ {os.path.basename(file)}: Too short")
                continue
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, sr)
                with open(tmp.name, 'rb') as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                os.unlink(tmp.name)
            
            features, samples, rate = audio_processor.process_audio_with_samples(audio_b64)
            result = voice_detector.detect(features, audio=samples, sr=rate, audio_bytes=audio_bytes)
            
            cls = result['classification']
            conf = result['confidenceScore']
            emoji = "🤖" if cls == "AI_GENERATED" else "👤"
            color = "#ff4444" if cls == "AI_GENERATED" else "#44ff88"
            
            if cls == "AI_GENERATED":
                ai_count += 1
            else:
                human_count += 1
            
            results.append(f"<div style='background:rgba(255,255,255,0.05);padding:10px;margin:5px 0;border-radius:8px;border-left:3px solid {color};'>"
                          f"<b>{emoji} {os.path.basename(file)}</b> → <span style='color:{color};'>{cls}</span> ({conf:.0%})</div>")
        except Exception as e:
            results.append(f"<div style='color:#ff4444;'>❌ {os.path.basename(file)}: {str(e)}</div>")
    
    summary = f"""
    <div style='background:rgba(102,126,234,0.2);padding:15px;border-radius:10px;margin-bottom:15px;text-align:center;'>
        <h3 style='color:#667eea;margin:0;'>📊 Batch Results</h3>
        <p style='margin:10px 0;'>🤖 AI: <b>{ai_count}</b> | 👤 Human: <b>{human_count}</b> | Total: <b>{len(files)}</b></p>
    </div>
    """
    return summary + "".join(results)


# Second demo for batch processing
batch_demo = gr.Interface(
    fn=process_batch,
    inputs=[
        gr.Files(label="📁 Upload Multiple Audio Files", file_types=["audio"]),
        gr.Dropdown(["English", "Tamil", "Hindi", "Malayalam", "Telugu"], value="English", label="Language")
    ],
    outputs=gr.HTML(label="Results"),
    title="📦 Batch Processing",
    description="Upload multiple audio files to analyze them all at once. Max 10 files."
)


# Combined app with tabs
combined_app = gr.TabbedInterface(
    [demo, batch_demo],
    ["🎤 Single Analysis", "📦 Batch Processing"],
    title="AI Voice Detection | India AI Impact Buildathon"
)


if __name__ == "__main__":
    combined_app.launch(server_name="0.0.0.0", server_port=7860)

