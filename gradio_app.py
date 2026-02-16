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


# ============ OPTION A: COMPARISON MODE ============
def compare_voices(audio1, audio2):
    """Compare two audio files side by side"""
    if audio1 is None or audio2 is None:
        return None, None, "<div style='text-align:center;padding:40px;color:#888;'>Upload two audio files to compare</div>"
    
    results = []
    spectrograms = []
    
    for i, audio in enumerate([audio1, audio2]):
        label = "Audio 1" if i == 0 else "Audio 2"
        try:
            if isinstance(audio, tuple):
                sr, audio_array = audio
                audio_array = audio_array.astype(np.float32)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)
            else:
                audio_array, sr = librosa.load(audio, sr=22050, mono=True)
            
            # Generate spectrogram
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
            ax.set_title(label, color='white', fontsize=12)
            ax.tick_params(colors='#888')
            for spine in ax.spines.values():
                spine.set_color('#333')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', dpi=100)
            plt.close()
            buf.seek(0)
            from PIL import Image
            spectrograms.append(Image.open(buf))
            
            # Convert to bytes for detection
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio_array, sr)
                with open(tmp.name, 'rb') as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                os.unlink(tmp.name)
            
            # Detect
            features, samples, rate = audio_processor.process_audio_with_samples(audio_b64)
            result = voice_detector.detect(features, audio=samples, sr=rate, audio_bytes=audio_bytes)
            
            cls = result['classification']
            conf = result['confidenceScore']
            emoji = "🤖" if cls == "AI_GENERATED" else "👤"
            color = "#ff4444" if cls == "AI_GENERATED" else "#44ff88"
            
            results.append({
                'label': label,
                'cls': cls,
                'conf': conf,
                'emoji': emoji,
                'color': color
            })
        except Exception as e:
            spectrograms.append(None)
            results.append({'label': label, 'cls': 'ERROR', 'conf': 0, 'emoji': '❌', 'color': '#888'})
    
    # Build comparison HTML
    r1, r2 = results[0], results[1]
    comparison_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px;">
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 4px solid {r1['color']}; text-align: center;">
            <h3 style="color: white; margin: 0 0 10px 0;">{r1['emoji']} Audio 1</h3>
            <div style="font-size: 24px; color: {r1['color']}; font-weight: bold;">{r1['cls'].replace('_', ' ')}</div>
            <div style="color: #888; margin-top: 5px;">Confidence: {r1['conf']:.0%}</div>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 4px solid {r2['color']}; text-align: center;">
            <h3 style="color: white; margin: 0 0 10px 0;">{r2['emoji']} Audio 2</h3>
            <div style="font-size: 24px; color: {r2['color']}; font-weight: bold;">{r2['cls'].replace('_', ' ')}</div>
            <div style="color: #888; margin-top: 5px;">Confidence: {r2['conf']:.0%}</div>
        </div>
    </div>
    <div style="text-align: center; padding: 15px; background: rgba(102,126,234,0.2); border-radius: 10px; margin-top: 10px;">
        <strong style="color: #667eea;">
            {'✅ Both voices are IDENTICAL type' if r1['cls'] == r2['cls'] else '⚠️ DIFFERENT voice types detected!'}
        </strong>
    </div>
    """
    
    return spectrograms[0], spectrograms[1], comparison_html


comparison_demo = gr.Interface(
    fn=compare_voices,
    inputs=[
        gr.Audio(label="🎤 Audio 1 (e.g., Human Voice)", sources=["upload", "microphone"]),
        gr.Audio(label="🤖 Audio 2 (e.g., AI Voice)", sources=["upload", "microphone"])
    ],
    outputs=[
        gr.Image(label="📊 Spectrogram 1"),
        gr.Image(label="📊 Spectrogram 2"),
        gr.HTML(label="🔍 Comparison Results")
    ],
    title="🔄 Voice Comparison Mode",
    description="Upload two audio files to compare them side-by-side. Great for comparing human vs AI voices!"
)


# ============ OPTION B: CONFIDENCE EXPLAINER ============
def explain_confidence(audio, language):
    """Generate detailed confidence explanation with visual breakdown"""
    if audio is None:
        return "<div style='text-align:center;padding:40px;color:#888;'>Upload audio to see confidence breakdown</div>"
    
    try:
        if isinstance(audio, tuple):
            sr, audio_array = audio
            audio_array = audio_array.astype(np.float32)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)
        else:
            audio_array, sr = librosa.load(audio, sr=22050, mono=True)
        
        # Convert to bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, sr)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            os.unlink(tmp.name)
        
        # Get detection with model scores
        features, samples, rate = audio_processor.process_audio_with_samples(audio_b64)
        result = voice_detector.detect(features, audio=samples, sr=rate, audio_bytes=audio_bytes)
        
        cls = result['classification']
        conf = result['confidenceScore']
        explanation = result.get('explanation', 'Analysis complete')
        model_scores = result.get('model_scores', {})
        
        # Build visual breakdown
        heuristic_score = model_scores.get('heuristic', 0)
        cnn_score = model_scores.get('cnn_mfcc', 0)
        transformer_score = model_scores.get('transformers', 0)
        
        def score_bar(label, score, weight, icon):
            color = "#ff4444" if score > 0.5 else "#44ff88"
            width = max(5, score * 100)
            return f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: white;">{icon} {label}</span>
                    <span style="color: #888;">Weight: {weight}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 30px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {color} 0%, {color}88 100%); height: 100%; width: {width}%; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; border-radius: 10px;">
                        <span style="color: white; font-weight: bold;">{score:.0%}</span>
                    </div>
                </div>
                <div style="color: #666; font-size: 0.8em; margin-top: 3px;">
                    {'Leans AI' if score > 0.5 else 'Leans Human'} - {'High' if abs(score - 0.5) > 0.3 else 'Medium' if abs(score - 0.5) > 0.15 else 'Low'} confidence
                </div>
            </div>
            """
        
        emoji = "🤖" if cls == "AI_GENERATED" else "👤"
        main_color = "#ff4444" if cls == "AI_GENERATED" else "#44ff88"
        
        html = f"""
        <div style="padding: 20px;">
            <!-- Final Result -->
            <div style="background: linear-gradient(135deg, {main_color}22 0%, {main_color}11 100%); padding: 25px; border-radius: 15px; text-align: center; border: 2px solid {main_color}; margin-bottom: 25px;">
                <div style="font-size: 48px;">{emoji}</div>
                <div style="font-size: 28px; color: {main_color}; font-weight: bold; margin: 10px 0;">{cls.replace('_', ' ')}</div>
                <div style="color: white; font-size: 20px;">Overall Confidence: {conf:.0%}</div>
            </div>
            
            <!-- Model Breakdown -->
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                <h3 style="color: #667eea; margin: 0 0 15px 0; text-align: center;">📊 Model Breakdown</h3>
                {score_bar("Wav2Vec2 Transformer", transformer_score, 60, "🤖")}
                {score_bar("CNN on MFCC", cnn_score, 35, "�")}
                {score_bar("Heuristic Analysis", heuristic_score, 30, "📊")}
            </div>
            
            <!-- Explanation -->
            <div style="background: rgba(102,126,234,0.15); padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h4 style="color: #667eea; margin: 0 0 10px 0;">🔬 Analysis Explanation</h4>
                <p style="color: #aaa; margin: 0; line-height: 1.6;">{explanation}</p>
            </div>
        </div>
        """
        return html
        
    except Exception as e:
        return f"<div style='color:#ff4444;padding:20px;'>Error: {str(e)}</div>"


explainer_demo = gr.Interface(
    fn=explain_confidence,
    inputs=[
        gr.Audio(label="🎤 Upload Audio", sources=["upload", "microphone"]),
        gr.Dropdown(["English", "Tamil", "Hindi", "Malayalam", "Telugu"], value="English", label="Language")
    ],
    outputs=gr.HTML(label="🔍 Confidence Breakdown"),
    title="🔬 Confidence Explainer",
    description="See exactly WHY the AI made its decision - visual breakdown of each model's contribution"
)


# ============ OPTION C: REAL-TIME WAVEFORM ============
def analyze_with_waveform(audio):
    """Show both waveform and spectrogram with analysis"""
    if audio is None:
        return None, None, "<div style='text-align:center;padding:40px;color:#888;'>Record or upload audio</div>"
    
    try:
        if isinstance(audio, tuple):
            sr, audio_array = audio
            audio_array = audio_array.astype(np.float32)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)
        else:
            audio_array, sr = librosa.load(audio, sr=22050, mono=True)
        
        # Generate Waveform
        fig1, ax1 = plt.subplots(figsize=(8, 2))
        fig1.patch.set_facecolor('#1a1a2e')
        ax1.set_facecolor('#1a1a2e')
        times = np.linspace(0, len(audio_array)/sr, len(audio_array))
        ax1.fill_between(times, audio_array, alpha=0.7, color='#667eea')
        ax1.plot(times, audio_array, color='#764ba2', linewidth=0.5)
        ax1.set_xlim(0, len(audio_array)/sr)
        ax1.set_ylim(-1, 1)
        ax1.set_xlabel('Time (s)', color='#888')
        ax1.set_ylabel('Amplitude', color='#888')
        ax1.set_title('🌊 Waveform', color='white')
        ax1.tick_params(colors='#888')
        for spine in ax1.spines.values():
            spine.set_color('#333')
        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', facecolor='#1a1a2e', dpi=100)
        plt.close()
        buf1.seek(0)
        from PIL import Image
        waveform_img = Image.open(buf1)
        
        # Generate Spectrogram
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
        ax2.set_title('📊 Mel-Spectrogram', color='white')
        ax2.tick_params(colors='#888')
        for spine in ax2.spines.values():
            spine.set_color('#333')
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', facecolor='#1a1a2e', dpi=100)
        plt.close()
        buf2.seek(0)
        spectrogram_img = Image.open(buf2)
        
        # Quick analysis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, sr)
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
        
        result_html = f"""
        <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%); padding: 25px; border-radius: 15px; text-align: center; border: 2px solid {color};">
            <div style="font-size: 48px;">{emoji}</div>
            <div style="font-size: 24px; color: {color}; font-weight: bold;">{cls.replace('_', ' ')}</div>
            <div style="color: white; margin-top: 5px;">Confidence: {conf:.0%}</div>
        </div>
        """
        
        return waveform_img, spectrogram_img, result_html
        
    except Exception as e:
        return None, None, f"<div style='color:#ff4444;'>Error: {str(e)}</div>"


waveform_demo = gr.Interface(
    fn=analyze_with_waveform,
    inputs=gr.Audio(label="🎤 Record or Upload Audio", sources=["upload", "microphone"]),
    outputs=[
        gr.Image(label="🌊 Waveform"),
        gr.Image(label="📊 Spectrogram"),
        gr.HTML(label="🔍 Result")
    ],
    title="🌊 Waveform Analysis",
    description="Visualize both waveform and spectrogram of your audio in real-time"
)


# ============ REAL-TIME STREAMING DETECTION ============
import time
import threading

# Global state for real-time detection
realtime_history = []

def realtime_detect(audio_stream, state):
    """Process streaming audio chunks for real-time detection"""
    if audio_stream is None:
        return (
            """<div style="text-align:center;padding:60px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #333;">
                <div style="font-size:64px;margin-bottom:15px;">🎙️</div>
                <h2 style="color:white;margin:0;">Click Record to Start Real-Time Detection</h2>
                <p style="color:#888;margin-top:10px;">Your microphone audio will be analyzed continuously</p>
            </div>""",
            state
        )
    
    try:
        # Handle Gradio audio input
        if isinstance(audio_stream, tuple):
            sr, audio_array = audio_stream
            audio_array = audio_array.astype(np.float32)
            if audio_array.dtype == np.int16:
                audio_array = audio_array / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array / 2147483648.0
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
        else:
            audio_array, sr = librosa.load(audio_stream, sr=22050, mono=True)
        
        duration = len(audio_array) / sr
        
        if duration < 0.5:
            return (
                """<div style="text-align:center;padding:40px;background:linear-gradient(135deg,#2a2a1e,#1e1e2e);border-radius:20px;border:2px solid #ffaa00;">
                    <div style="font-size:48px;">⏳</div>
                    <h3 style="color:#ffcc00;">Recording... Keep speaking!</h3>
                    <p style="color:#888;">Need at least 0.5 seconds of audio for analysis</p>
                    <div style="margin-top:15px;">
                        <div style="height:8px;background:#333;border-radius:4px;overflow:hidden;">
                            <div style="width:""" + f"{min(duration/0.5*100, 100):.0f}" + """%;height:100%;background:linear-gradient(90deg,#ffaa00,#ffcc00);border-radius:4px;transition:width 0.3s;"></div>
                        </div>
                    </div>
                </div>""",
                state
            )
        
        # Convert to bytes for detection
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, sr)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            os.unlink(tmp.name)
        
        # Run detection
        features, audio_samples, rate = audio_processor.process_audio_with_samples(audio_base64)
        result = voice_detector.detect(
            features=features,
            audio=audio_samples,
            sr=rate,
            audio_bytes=audio_bytes
        )
        
        classification = result['classification']
        confidence = result['confidenceScore']
        is_ai = classification == "AI_GENERATED"
        
        # Update history
        timestamp = time.strftime("%H:%M:%S")
        if state is None:
            state = []
        state.append({
            'time': timestamp,
            'cls': classification,
            'conf': confidence,
            'duration': duration
        })
        # Keep last 10 entries
        state = state[-10:]
        
        # Build live dashboard
        emoji = "🤖" if is_ai else "👤"
        color = "#ff4444" if is_ai else "#00ff88"
        bg_gradient = "linear-gradient(135deg, #ff444422, #ff000011)" if is_ai else "linear-gradient(135deg, #00ff8822, #00aa5511)"
        label = "AI GENERATED" if is_ai else "HUMAN VOICE"
        pulse = "animation: pulse 1s infinite;" if is_ai else ""
        
        # History rows
        history_html = ""
        for entry in reversed(state[-5:]):
            h_color = "#ff4444" if entry['cls'] == "AI_GENERATED" else "#00ff88"
            h_emoji = "🤖" if entry['cls'] == "AI_GENERATED" else "👤"
            h_label = "AI" if entry['cls'] == "AI_GENERATED" else "HUMAN"
            history_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 15px;background:rgba(255,255,255,0.05);border-radius:8px;margin-bottom:5px;border-left:3px solid {h_color};">
                <span style="color:#888;">{entry['time']}</span>
                <span style="color:{h_color};font-weight:bold;">{h_emoji} {h_label}</span>
                <span style="color:white;">{entry['conf']:.0%}</span>
                <span style="color:#888;">{entry['duration']:.1f}s</span>
            </div>"""
        
        dashboard = f"""
        <style>
            @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.7; }} }}
            @keyframes liveDot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
        </style>
        <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border-radius:20px;overflow:hidden;border:2px solid {color}33;">
            <!-- Live indicator -->
            <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:rgba(0,0,0,0.3);">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:10px;height:10px;background:#ff0000;border-radius:50%;animation:liveDot 1s infinite;"></div>
                    <span style="color:white;font-weight:bold;">LIVE DETECTION</span>
                </div>
                <span style="color:#888;">Audio: {duration:.1f}s</span>
            </div>
            
            <!-- Main result -->
            <div style="text-align:center;padding:30px;{bg_gradient};{pulse}">
                <div style="font-size:64px;margin-bottom:10px;">{emoji}</div>
                <div style="font-size:28px;color:{color};font-weight:bold;letter-spacing:2px;">{label}</div>
                <div style="margin-top:15px;">
                    <div style="background:rgba(0,0,0,0.4);height:20px;border-radius:10px;overflow:hidden;max-width:300px;margin:0 auto;border:1px solid rgba(255,255,255,0.1);">
                        <div style="width:{confidence*100}%;height:100%;background:linear-gradient(90deg,{color},{color}88);border-radius:10px;transition:width 0.5s;"></div>
                    </div>
                    <div style="color:white;font-size:36px;font-weight:bold;margin-top:8px;">{confidence*100:.1f}%</div>
                </div>
            </div>
            
            <!-- History -->
            <div style="padding:15px 20px;">
                <h4 style="color:#888;margin:0 0 10px 0;font-size:13px;text-transform:uppercase;letter-spacing:1px;">Detection History</h4>
                {history_html}
            </div>
        </div>
        """
        
        return dashboard, state
        
    except Exception as e:
        return (
            f"""<div style="text-align:center;padding:30px;background:#2a1a1a;border-radius:15px;border:2px solid #ff4444;">
                <div style="font-size:36px;">⚠️</div>
                <p style="color:#ff6666;">Error: {str(e)}</p>
            </div>""",
            state
        )


# Build Real-Time tab with Blocks
with gr.Blocks(css=CUSTOM_CSS) as realtime_demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px;background:linear-gradient(90deg,#ff416c,#ff4b2b);border-radius:15px;margin-bottom:20px;">
        <h2 style="color:white;margin:0;">🔴 Real-Time Voice Detection</h2>
        <p style="color:rgba(255,255,255,0.9);margin:5px 0 0 0;">Speak into your microphone — AI analyzes your voice continuously as you talk</p>
    </div>
    """)
    
    state = gr.State({"audio_buffer": None, "sr": None, "history": [], "chunk_count": 0})
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div style="padding:10px;background:rgba(255,65,108,0.1);border-radius:10px;border:1px solid rgba(255,65,108,0.3);margin-bottom:10px;">
                <p style="color:#ff8a9e;margin:0;font-size:14px;">🎙️ <b>How it works:</b> Click the mic button and start speaking. The AI will <b>continuously analyze</b> your voice in real-time as you talk — no need to stop!</p>
            </div>""")
            audio_input = gr.Audio(
                label="🎙️ Speak Now — Live Analysis",
                sources=["microphone"],
                type="numpy",
                streaming=True
            )
        
        with gr.Column(scale=2):
            result_html = gr.HTML(
                value="""<div style="text-align:center;padding:60px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #333;">
                    <div style="font-size:64px;margin-bottom:15px;">🎙️</div>
                    <h2 style="color:white;margin:0;">Click the Mic to Start Live Detection</h2>
                    <p style="color:#888;margin-top:10px;">AI will analyze your voice continuously as you speak</p>
                </div>"""
            )
    
    def stream_detect(audio_chunk, current_state):
        """Process streaming audio chunks in real-time while user speaks"""
        if audio_chunk is None:
            return (
                """<div style="text-align:center;padding:60px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #333;">
                    <div style="font-size:64px;margin-bottom:15px;">🎙️</div>
                    <h2 style="color:white;margin:0;">Click the Mic to Start Live Detection</h2>
                    <p style="color:#888;margin-top:10px;">AI will analyze your voice continuously as you speak</p>
                </div>""",
                current_state
            )
        
        try:
            sr, chunk_array = audio_chunk
            chunk_array = chunk_array.astype(np.float32)
            if chunk_array.max() > 1.0:
                chunk_array = chunk_array / 32768.0
            if len(chunk_array.shape) > 1:
                chunk_array = chunk_array.mean(axis=1)
            
            # Accumulate audio in buffer
            if current_state["audio_buffer"] is None:
                current_state["audio_buffer"] = chunk_array
                current_state["sr"] = sr
            else:
                current_state["audio_buffer"] = np.concatenate([current_state["audio_buffer"], chunk_array])
            
            current_state["chunk_count"] = current_state.get("chunk_count", 0) + 1
            
            buffer = current_state["audio_buffer"]
            buffer_sr = current_state["sr"]
            duration = len(buffer) / buffer_sr
            
            # Need at least 2 seconds of audio for reliable detection
            if duration < 2.0:
                progress = min(duration / 2.0 * 100, 100)
                return (
                    f"""<div style="text-align:center;padding:40px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #ffaa00;">
                        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:15px;">
                            <div style="width:10px;height:10px;background:#ff0000;border-radius:50%;animation:liveDot 1s infinite;"></div>
                            <span style="color:white;font-weight:bold;">LISTENING...</span>
                        </div>
                        <div style="font-size:48px;">🎤</div>
                        <h3 style="color:#ffcc00;">Keep speaking... ({duration:.1f}s)</h3>
                        <div style="max-width:300px;margin:15px auto;">
                            <div style="height:10px;background:#333;border-radius:5px;overflow:hidden;">
                                <div style="width:{progress}%;height:100%;background:linear-gradient(90deg,#ffaa00,#ffcc00);border-radius:5px;transition:width 0.3s;"></div>
                            </div>
                        </div>
                        <p style="color:#888;">Analyzing after 2 seconds of audio</p>
                        <style>@keyframes liveDot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}</style>
                    </div>""",
                    current_state
                )
            
            # Check if audio is too quiet (silence) - skip analysis to avoid false positives
            rms = np.sqrt(np.mean(buffer[-int(buffer_sr):] ** 2))
            if rms < 0.01:
                silence_html = current_state.get("last_html", None)
                if silence_html:
                    return silence_html, current_state
                return (
                    f"""<div style="text-align:center;padding:40px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #555;">
                        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:15px;">
                            <div style="width:10px;height:10px;background:#ff0000;border-radius:50%;animation:liveDot 1s infinite;"></div>
                            <span style="color:white;font-weight:bold;">LISTENING...</span>
                        </div>
                        <div style="font-size:48px;">🔇</div>
                        <h3 style="color:#888;">Speak louder — audio too quiet to analyze</h3>
                        <p style="color:#555;">Volume: {rms:.4f} (need > 0.01)</p>
                        <style>@keyframes liveDot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}</style>
                    </div>""",
                    current_state
                )
            
            # Only run detection every 5 chunks to give enough audio between analyses
            if current_state["chunk_count"] % 5 != 0:
                # Return previous result if available
                if current_state.get("last_html"):
                    return current_state["last_html"], current_state
                return (
                    f"""<div style="text-align:center;padding:30px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;border:2px solid #667eea;">
                        <div style="display:flex;align-items:center;justify-content:center;gap:8px;">
                            <div style="width:10px;height:10px;background:#ff0000;border-radius:50%;animation:liveDot 1s infinite;"></div>
                            <span style="color:white;font-weight:bold;">PROCESSING... ({duration:.1f}s)</span>
                        </div>
                        <style>@keyframes liveDot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}</style>
                    </div>""",
                    current_state
                )
            
            # Use last 5 seconds of audio for analysis (sliding window)
            analysis_samples = int(min(5.0, duration) * buffer_sr)
            analysis_audio = buffer[-analysis_samples:]
            
            # Resample if needed
            if buffer_sr != 22050:
                analysis_audio = librosa.resample(analysis_audio, orig_sr=buffer_sr, target_sr=22050)
                analysis_sr = 22050
            else:
                analysis_sr = buffer_sr
            
            # Convert to bytes for detection
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, analysis_audio, analysis_sr)
                with open(tmp.name, 'rb') as f:
                    audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                os.unlink(tmp.name)
            
            # Run detection
            features, audio_samples_out, rate = audio_processor.process_audio_with_samples(audio_base64)
            result = voice_detector.detect(
                features=features,
                audio=audio_samples_out,
                sr=rate,
                audio_bytes=audio_bytes
            )
            
            classification = result['classification']
            confidence = result['confidenceScore']
            
            # Streaming-specific adjustment: mic audio has compression artifacts
            # that inflate transformer + heuristic scores falsely.
            # The CNN model is most reliable for raw mic audio.
            # If CNN says human but transformer disagrees (due to mic artifacts), trust CNN.
            model_scores = result.get('model_scores', {})
            tf_score = model_scores.get('transformers', 0)
            cnn_score = model_scores.get('cnn_mfcc', 0)
            heuristic_score = model_scores.get('heuristic', 0)
            
            if classification == "AI_GENERATED":
                # In streaming mode, CNN is most reliable since it analyzes spectral
                # patterns that aren't affected by streaming artifacts.
                # If CNN says human (< 0.3) but transformer says AI, it's likely a false positive.
                if cnn_score < 0.30:
                    classification = "HUMAN"
                    confidence = max(0.55, 1.0 - cnn_score)
                # Also if only heuristic is high but both DL models disagree
                elif tf_score < 0.3 and cnn_score < 0.3:
                    classification = "HUMAN"
                    confidence = 0.70
            
            is_ai = classification == "AI_GENERATED"
            
            # Update history
            timestamp = time.strftime("%H:%M:%S")
            history = current_state.get("history", [])
            history.append({
                'time': timestamp,
                'cls': classification,
                'conf': confidence,
                'duration': duration
            })
            current_state["history"] = history[-10:]
            
            # Keep buffer manageable (last 5 seconds only)
            max_samples = int(5.0 * buffer_sr)
            if len(buffer) > max_samples:
                current_state["audio_buffer"] = buffer[-max_samples:]
            
            # Build live dashboard
            emoji = "🤖" if is_ai else "👤"
            color = "#ff4444" if is_ai else "#00ff88"
            label = "AI GENERATED" if is_ai else "HUMAN VOICE"
            pulse = "animation: pulse 1s infinite;" if is_ai else ""
            alert_bg = "#ff444422" if is_ai else "#00ff8822"
            
            # History rows
            history_html = ""
            for entry in reversed(current_state["history"][-5:]):
                h_color = "#ff4444" if entry['cls'] == "AI_GENERATED" else "#00ff88"
                h_emoji = "🤖" if entry['cls'] == "AI_GENERATED" else "👤"
                h_label = "AI" if entry['cls'] == "AI_GENERATED" else "HUMAN"
                history_html += f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 15px;background:rgba(255,255,255,0.05);border-radius:8px;margin-bottom:5px;border-left:3px solid {h_color};">
                    <span style="color:#888;">{entry['time']}</span>
                    <span style="color:{h_color};font-weight:bold;">{h_emoji} {h_label}</span>
                    <span style="color:white;">{entry['conf']:.0%}</span>
                    <span style="color:#888;">{entry['duration']:.1f}s</span>
                </div>"""
            
            dashboard = f"""
            <style>
                @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.7; }} }}
                @keyframes liveDot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
            </style>
            <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border-radius:20px;overflow:hidden;border:2px solid {color}33;">
                <!-- Live indicator -->
                <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:rgba(0,0,0,0.3);">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="width:10px;height:10px;background:#ff0000;border-radius:50%;animation:liveDot 1s infinite;"></div>
                        <span style="color:white;font-weight:bold;">🔴 LIVE DETECTION</span>
                    </div>
                    <span style="color:#888;">Buffer: {duration:.1f}s | Analyses: {len(current_state['history'])}</span>
                </div>
                
                <!-- Main result -->
                <div style="text-align:center;padding:30px;background:{alert_bg};{pulse}">
                    <div style="font-size:64px;margin-bottom:10px;">{emoji}</div>
                    <div style="font-size:28px;color:{color};font-weight:bold;letter-spacing:2px;">{label}</div>
                    <div style="margin-top:15px;">
                        <div style="background:rgba(0,0,0,0.4);height:20px;border-radius:10px;overflow:hidden;max-width:300px;margin:0 auto;border:1px solid rgba(255,255,255,0.1);">
                            <div style="width:{confidence*100}%;height:100%;background:linear-gradient(90deg,{color},{color}88);border-radius:10px;transition:width 0.5s;"></div>
                        </div>
                        <div style="color:white;font-size:36px;font-weight:bold;margin-top:8px;">{confidence*100:.1f}%</div>
                    </div>
                </div>
                
                <!-- History -->
                <div style="padding:15px 20px;">
                    <h4 style="color:#888;margin:0 0 10px 0;font-size:13px;text-transform:uppercase;letter-spacing:1px;">Live Detection History</h4>
                    {history_html if history_html else '<p style="color:#555;text-align:center;">Results will appear here as you speak...</p>'}
                </div>
            </div>
            """
            
            current_state["last_html"] = dashboard
            return dashboard, current_state
            
        except Exception as e:
            return (
                f"""<div style="text-align:center;padding:30px;background:#2a1a1a;border-radius:15px;border:2px solid #ff4444;">
                    <div style="font-size:36px;">⚠️</div>
                    <p style="color:#ff6666;">Error: {str(e)}</p>
                </div>""",
                current_state
            )
    
    audio_input.stream(
        fn=stream_detect,
        inputs=[audio_input, state],
        outputs=[result_html, state]
    )


# ============ DEVELOPER API TAB ============
API_BASE = "https://gaurav00107-ai-voice-detection.hf.space"

with gr.Blocks(css=CUSTOM_CSS) as api_demo:
    gr.HTML(f"""
    <div style="text-align:center;padding:20px;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:15px;margin-bottom:20px;">
        <h2 style="color:white;margin:0;">🔑 Developer API</h2>
        <p style="color:rgba(255,255,255,0.9);margin:5px 0 0 0;">Integrate AI Voice Detection into your app in minutes</p>
    </div>
    
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:25px;">
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;text-align:center;border:1px solid #333;">
            <div style="font-size:32px;">⚡</div>
            <h3 style="color:white;margin:8px 0 4px;">REST API</h3>
            <p style="color:#888;margin:0;font-size:13px;">Simple POST request with base64 audio</p>
        </div>
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;text-align:center;border:1px solid #333;">
            <div style="font-size:32px;">🌍</div>
            <h3 style="color:white;margin:8px 0 4px;">5 Languages</h3>
            <p style="color:#888;margin:0;font-size:13px;">Tamil, Hindi, English, Malayalam, Telugu</p>
        </div>
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;text-align:center;border:1px solid #333;">
            <div style="font-size:32px;">🆓</div>
            <h3 style="color:white;margin:8px 0 4px;">Free Tier</h3>
            <p style="color:#888;margin:0;font-size:13px;">100 requests/day — no API key needed</p>
        </div>
    </div>
    """)
    
    gr.HTML(f"""
    <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border-radius:15px;padding:25px;border:1px solid #333;margin-bottom:20px;">
        <h3 style="color:white;margin:0 0 5px;">📡 API Endpoint</h3>
        <div style="background:#000;padding:12px 18px;border-radius:8px;font-family:monospace;display:flex;align-items:center;justify-content:space-between;margin-top:10px;">
            <span style="color:#00ff88;font-size:15px;">POST {API_BASE}/</span>
        </div>
        <p style="color:#888;margin:10px 0 0;font-size:13px;">
            📖 Interactive Swagger Docs: <a href="{API_BASE}/docs" target="_blank" style="color:#667eea;">{API_BASE}/docs</a>
        </p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🐍 Python"):
            gr.Code(
                value=f'''import requests, base64

# Read your audio file
with open("audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Call the API
response = requests.post(
    "{API_BASE}/",
    json={{
        "audioBase64": audio_b64,
        "language": "english",
        "audioFormat": "mp3"
    }}
)

result = response.json()
print(f"Verdict: {{result['classification']}}")
print(f"Confidence: {{result['confidenceScore']}}%")
print(f"Explanation: {{result['explanation']}}")''',
                language="python",
                label="Python Example"
            )
        
        with gr.Tab("🌀 cURL"):
            gr.Code(
                value=f'''# Encode audio to base64
AUDIO_B64=$(base64 -i audio.mp3)

# Call the API
curl -X POST "{API_BASE}/" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "audioBase64": "'$AUDIO_B64'",
    "language": "english",
    "audioFormat": "mp3"
  }}'
''',
                language="shell",
                label="cURL Example"
            )
        
        with gr.Tab("🟨 JavaScript"):
            gr.Code(
                value=f'''// Browser: Read file and call API
const file = document.getElementById("audioInput").files[0];
const reader = new FileReader();

reader.onload = async () => {{
  const base64 = reader.result.split(",")[1];
  
  const response = await fetch("{API_BASE}/", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{
      audioBase64: base64,
      language: "english",
      audioFormat: "mp3"
    }})
  }});
  
  const result = await response.json();
  console.log(`Verdict: ${{result.classification}}`);
  console.log(`Confidence: ${{result.confidenceScore}}%`);
}};

reader.readAsDataURL(file);''',
                language="javascript",
                label="JavaScript Example"
            )
        
        with gr.Tab("📦 Batch API"):
            gr.Code(
                value=f'''import requests, base64, glob

# Batch analyze multiple files
files = glob.glob("audio_samples/*.mp3")
results = []

for filepath in files:
    with open(filepath, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    
    resp = requests.post(
        "{API_BASE}/api/voice-detection",
        json={{
            "audioBase64": audio_b64,
            "language": "english",
            "audioFormat": "mp3"
        }}
    )
    result = resp.json()
    results.append({{
        "file": filepath,
        "verdict": result["classification"],
        "confidence": result["confidenceScore"]
    }})
    print(f"{{filepath}}: {{result['classification']}} ({{result['confidenceScore']}}%)")

# Summary
ai_count = sum(1 for r in results if r["verdict"] == "AI_GENERATED")
print(f"\\n📊 Total: {{len(results)}} files | AI: {{ai_count}} | Human: {{len(results)-ai_count}}")''',
                language="python",
                label="Batch Processing Example"
            )
    
    gr.HTML(f"""
    <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border-radius:15px;padding:25px;border:1px solid #333;margin-top:20px;">
        <h3 style="color:white;margin:0 0 15px;">📋 Response Format</h3>
        <pre style="background:#000;padding:15px;border-radius:8px;color:#e0e0e0;overflow-x:auto;font-size:13px;"><code style="color:#e0e0e0;">{{
  "classification": "AI_GENERATED" | "HUMAN",
  "confidenceScore": 0.92,
  "explanation": "AI voice detected: synthetic pitch consistency, deep learning detected synthetic artifacts",
  "method": "heuristic+cnn_mfcc+transformers",
  "model_scores": {{
    "heuristic": 0.89,
    "cnn_mfcc": 0.85,
    "transformers": 0.97
  }}
}}</code></pre>
    </div>
    
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:20px;">
        <div style="background:linear-gradient(135deg,#1a2a1a,#1a3a1a);padding:20px;border-radius:12px;border:1px solid #2a4a2a;">
            <h4 style="color:#00ff88;margin:0 0 10px;">✅ Use Cases</h4>
            <ul style="color:#ccc;margin:0;padding-left:20px;font-size:13px;line-height:1.8;">
                <li>Banking — Voice authentication fraud detection</li>
                <li>Call Centers — Screen incoming calls for AI deepfakes</li>
                <li>Media — Verify audio evidence authenticity</li>
                <li>Social Media — Flag AI-generated voice content</li>
                <li>Insurance — Detect fraudulent voice claims</li>
            </ul>
        </div>
        <div style="background:linear-gradient(135deg,#1a1a2e,#2a1a3e);padding:20px;border-radius:12px;border:1px solid #3a2a4a;">
            <h4 style="color:#667eea;margin:0 0 10px;">📊 API Specs</h4>
            <ul style="color:#ccc;margin:0;padding-left:20px;font-size:13px;line-height:1.8;">
                <li>Response time: &lt; 3 seconds</li>
                <li>Max audio: 30 seconds per request</li>
                <li>Formats: MP3, WAV, OGG, FLAC, M4A</li>
                <li>Languages: Tamil, Hindi, English, Malayalam, Telugu</li>
                <li>Auth: API key (optional, free tier available)</li>
            </ul>
        </div>
    </div>
    """)


# Combined app with all tabs
combined_app = gr.TabbedInterface(
    [demo, batch_demo, comparison_demo, explainer_demo, waveform_demo, realtime_demo, api_demo],
    ["🎤 Single", "📦 Batch", "🔄 Compare", "🔬 Explainer", "🌊 Waveform", "🔴 Real-Time", "🔑 API"],
    title="AI Voice Detection | India AI Impact Buildathon"
)


# Mount Gradio app on FastAPI app to expose API routes
from app.main import app as fastapi_app
import uvicorn

# Mount at root path so HuggingFace Spaces can display the UI directly
app = gr.mount_gradio_app(fastapi_app, combined_app, path="/")


if __name__ == "__main__":
    # Use uvicorn to run the FastAPI app (which now includes Gradio)
    uvicorn.run(app, host="0.0.0.0", port=7860)
