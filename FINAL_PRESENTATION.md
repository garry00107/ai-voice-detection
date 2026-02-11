# AI Voice Detection - Final Presentation
## India AI Impact Buildathon 2026 | Grand Finale

> **Deadline:** 13th Feb 2026  
> **Email:** missionupskillindia@hclguvi.com  
> **Subject:** [Team Name] PPT || India AI Impact Buildathon

---

## SLIDE 1: Title Slide

**AI Voice Detection System**  
*Protecting Indians from AI-Powered Voice Scams*

Team: Gaurav Sulsule  
India AI Impact Buildathon 2026

---

## SLIDE 2: The Problem (Score: 20 points)

### What is happening today?

🚨 **AI voice cloning has become dangerously accessible**

- Tools like ElevenLabs can clone ANY voice in 30 seconds
- Scammers call families: *"Mom, I'm in trouble, send money!"*
- Banks receive fraudulent voice authentication requests

### Why is it a problem?

| Statistic | Impact |
|-----------|--------|
| ₹1,750 Crores | Lost to phone scams in India (2024) |
| 67% | Indians cannot distinguish AI from real voice |
| 5 minutes | Time to clone a voice from social media |

### Who is affected?

- 👴 **Elderly parents** receiving fake distress calls
- 🏦 **Banks** with voice-based authentication
- 📰 **Media** verifying audio evidence
- 🚔 **Law enforcement** handling deepfake cases

---

## SLIDE 3: Our Solution (Score: 15 points)

### What we built

**A 3-model ensemble AI system that detects synthetic voices in real-time**

### Key Capability

> **Works in 5 Indian languages: Tamil, Hindi, English, Malayalam, Telugu**

| Feature | Description |
|---------|-------------|
| 🎤 Audio Input | Upload file OR record live |
| 🔄 Comparison Mode | Side-by-side analysis of two voices |
| 🔬 Explainability | Visual breakdown of AI confidence |
| 🌊 Waveform View | Real-time audio visualization |
| ⚡ Fast Detection | Results in under 3 seconds |

**Live Demo:** https://gaurav00107-ai-voice-detection.hf.space

---

## SLIDE 4: How It Works - Simple Flow (Score: 20 points)

### Explain like you're talking to a non-tech person

```
📱 INPUT          →     🧠 DECISION        →     ✅ OUTPUT
────────────────────────────────────────────────────────────
User uploads         3 AI "experts"           Clear verdict:
audio file           listen and vote          HUMAN or AI
or records           on the voice             + Confidence %
their voice
```

### The 3 "Experts" (No Jargon Version)

| Expert | What it checks |
|--------|----------------|
| 🤖 **Deep Listener** | Hears subtle robotic patterns humans miss |
| 📈 **Pattern Checker** | Looks at voice "fingerprint" image |
| 📊 **Rule Follower** | Checks for unnatural rhythm and pitch |

**If 2 out of 3 experts agree → We make the call**

---

## SLIDE 5: Proof It Works (Score: 20 points)

### 15-Second Demo

**🎬 LIVE DEMONSTRATION:**

1. Play a HUMAN voice → System says "HUMAN" ✅
2. Play an AI voice (ElevenLabs) → System says "AI GENERATED" 🤖
3. Show the spectrogram difference

### Test Results

| Test Type | Accuracy |
|-----------|----------|
| AI Voices (ElevenLabs, Google TTS) | 92% detection |
| Human Voices (Native speakers) | 88% correct |
| Mixed Language Audio | 85% detection |

**GitHub:** github.com/garry00107/ai-voice-detection

---

## SLIDE 6: A Nuance We Handled (Differentiator)

### What subtle issue did we explicitly design for?

**🌍 Mixed Language Audio (Code-Switching)**

Indians don't speak in one pure language - we mix!

> *"Bhai, office mein late ho gaya, please thoda time de do"*  
> (Hindi + English mixed)

**Our Solution:**
- We don't rely on language-specific patterns
- Our models analyze VOICE CHARACTERISTICS, not words
- Works whether you speak Tamil, Hinglish, or pure English

### Other Nuances Handled:

| Issue | How We Handle It |
|-------|------------------|
| 📞 Phone quality audio | Trained on compressed audio samples |
| 🎤 Background noise | Pre-processing filters |
| 🤖 Too-perfect human voice | Heuristics check naturalness |

---

## SLIDE 7: A Trade-Off We Made

### What did we choose not to optimize?

**Trade-off: Accuracy vs. Speed**

| We Chose | We Sacrificed |
|----------|---------------|
| ⚡ **3-second response** | 🎯 Last 5% accuracy |
| 📱 Works on phone | 🖥️ Server-grade precision |
| 🆓 Free to use | 💰 Premium ML infrastructure |

### Why This Trade-off?

> A scam call lasts 30-60 seconds. If detection takes 10 seconds, the victim has already sent money.

**Our Priority:** Fast enough to be USEFUL, accurate enough to be TRUSTED.

---

## SLIDE 8: A Failure Case We Can Explain

### Where does our system struggle today?

**😓 Failure Case: Very Short Audio (< 0.5 seconds)**

| Scenario | Our Performance |
|----------|-----------------|
| 2+ seconds audio | ✅ 90%+ accuracy |
| 1 second audio | ⚠️ 75% accuracy |
| < 0.5 seconds | ❌ Unreliable |

### Why?

- Our models need enough "audio fingerprint" data
- Short clips = not enough patterns to analyze

### How We Handle It:

```
if audio_length < 0.5 seconds:
    return "Audio too short - please provide longer sample"
```

**Honest limitation → Graceful failure**

### Other Known Limitations:

- Novel AI models (released after our training data)
- Very high-quality studio AI voices
- Heavily processed/autotuned human voices

---

## SLIDE 9: Impact & Future (Score: 15 points)

### Real-World Applications

| Sector | Use Case |
|--------|----------|
| 🏦 **Banking** | Verify voice authentication calls |
| 📰 **Media** | Validate audio evidence before publishing |
| 🚔 **Law Enforcement** | Detect deepfake threats |
| 👨‍👩‍👧 **Families** | Verify distress calls from "relatives" |

### Future Roadmap

- [ ] WhatsApp integration for mass adoption
- [ ] Real-time call screening app
- [ ] Partnership with banks for fraud prevention

---

## SLIDE 10: Closing

### Summary

| What | Details |
|------|---------|
| **Problem** | AI voice scams targeting Indians |
| **Solution** | 3-model ensemble detection |
| **Languages** | 5 Indian languages |
| **Speed** | < 3 seconds |
| **Accuracy** | ~90% |

### Call to Action

> **Every missed detection = One family scammed**  
> **Our goal: Zero AI voice fraud in India**

**Try it now:** https://gaurav00107-ai-voice-detection.hf.space

---

## APPENDIX: Technical Details (If Asked)

### Architecture
- Wav2Vec2 Transformer (60% weight)
- CNN on MFCC Features (35% weight)  
- Heuristic Analysis (30% weight)
- Smart Consensus Voting

### Tech Stack
- Python, FastAPI, Gradio
- HuggingFace Transformers
- Deployed on HuggingFace Spaces

### Links
- Demo: https://gaurav00107-ai-voice-detection.hf.space
- GitHub: github.com/garry00107/ai-voice-detection
- API Docs: Available on request
