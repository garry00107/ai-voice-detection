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


# Combined app with all tabs
combined_app = gr.TabbedInterface(
    [demo, batch_demo, comparison_demo, explainer_demo, waveform_demo],
    ["🎤 Single", "📦 Batch", "🔄 Compare", "🔬 Explainer", "🌊 Waveform"],
    title="AI Voice Detection | India AI Impact Buildathon"
)


# Mount Gradio app on FastAPI app to expose API routes
from app.main import app as fastapi_app
import uvicorn

# Mount at /gradio to allow API routes to work
app = gr.mount_gradio_app(fastapi_app, combined_app, path="/gradio")


if __name__ == "__main__":
    # Use uvicorn to run the FastAPI app (which now includes Gradio)
    uvicorn.run(app, host="0.0.0.0", port=7860)
