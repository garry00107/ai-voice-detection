"""
Audio Processing Module
Handles Base64 decoding, audio conversion, and MFCC feature extraction
"""
import base64
import io
import tempfile
import os
from typing import Dict, Any, Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydub import AudioSegment


class AudioProcessor:
    """Processes audio files and extracts features for voice detection."""
    
    # Audio processing parameters
    SAMPLE_RATE = 22050  # Standard sample rate for librosa
    N_MFCC = 13  # Number of MFCC coefficients
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Hop length for STFT
    
    def __init__(self):
        """Initialize the audio processor."""
        pass
    
    def decode_base64_audio(self, audio_base64: str) -> bytes:
        """
        Decode Base64 encoded audio to bytes.
        
        Args:
            audio_base64: Base64 encoded audio string
            
        Returns:
            Audio data as bytes
        """
        try:
            # Handle potential data URI prefix
            if ',' in audio_base64:
                audio_base64 = audio_base64.split(',')[1]
            
            return base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64 audio: {str(e)}")
    
    def convert_audio_to_samples(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Convert audio bytes (MP3 or WAV) to numpy array.
        Uses multiple fallback methods for robustness.
        
        Args:
            audio_bytes: Audio as bytes (MP3 or WAV)
            
        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        temp_path = None
        try:
            # Detect format from file signature
            is_wav = audio_bytes[:4] == b'RIFF' or audio_bytes[:4] == b'riff'
            
            # Determine suffix
            suffix = '.wav' if is_wav else '.mp3'
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Method 1: Try librosa directly (works for many formats)
            try:
                y, sr = librosa.load(temp_path, sr=self.SAMPLE_RATE, mono=True)
                return y, sr
            except Exception as e1:
                pass
            
            # Method 2: Try soundfile
            try:
                import soundfile as sf
                y, sr = sf.read(temp_path)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)  # Convert to mono
                # Resample if needed
                if sr != self.SAMPLE_RATE:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.SAMPLE_RATE)
                    sr = self.SAMPLE_RATE
                return y.astype(np.float32), sr
            except Exception as e2:
                pass
            
            # Method 3: Try pydub (requires ffmpeg)
            try:
                audio = AudioSegment.from_file(temp_path)
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                y, sr = librosa.load(wav_buffer, sr=self.SAMPLE_RATE, mono=True)
                return y, sr
            except Exception as e3:
                raise ValueError(f"Failed to convert audio with all methods. Last error: {str(e3)}")
                
        except Exception as e:
            raise ValueError(f"Failed to convert audio: {str(e)}")
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract audio features for voice detection.
        
        Features extracted:
        - MFCCs (mean, std, min, max for each coefficient)
        - Delta MFCCs
        - Delta-delta MFCCs
        - Spectral centroid
        - Spectral rolloff
        - Zero crossing rate
        - RMS energy
        - Pitch statistics
        
        Args:
            audio: Audio samples as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Ensure audio is long enough
        if len(audio) < self.N_FFT:
            # Pad short audio
            audio = np.pad(audio, (0, self.N_FFT - len(audio)), mode='constant')
        
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )
        
        # MFCC statistics
        for i in range(self.N_MFCC):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            features[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
            features[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))
        
        # 2. Delta MFCCs (velocity)
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(self.N_MFCC):
            features[f'delta_mfcc_{i}_mean'] = float(np.mean(delta_mfccs[i]))
            features[f'delta_mfcc_{i}_std'] = float(np.std(delta_mfccs[i]))
        
        # 3. Delta-delta MFCCs (acceleration)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(self.N_MFCC):
            features[f'delta2_mfcc_{i}_mean'] = float(np.mean(delta2_mfccs[i]))
            features[f'delta2_mfcc_{i}_std'] = float(np.std(delta2_mfccs[i]))
        
        # 4. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        # 5. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.HOP_LENGTH)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=self.HOP_LENGTH)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 8. Pitch/F0 analysis using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            # Filter out NaN values (unvoiced regions)
            f0_voiced = f0[~np.isnan(f0)]
            if len(f0_voiced) > 0:
                features['pitch_mean'] = float(np.mean(f0_voiced))
                features['pitch_std'] = float(np.std(f0_voiced))
                features['pitch_range'] = float(np.max(f0_voiced) - np.min(f0_voiced))
                features['voiced_ratio'] = float(np.sum(voiced_flag) / len(voiced_flag))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
                features['voiced_ratio'] = 0.0
        except Exception:
            # Fallback if pitch extraction fails
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
            features['voiced_ratio'] = 0.0
        
        # 9. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # 10. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))
        
        # 11. Tonnetz (tonal centroid features)
        try:
            harmonic = librosa.effects.harmonic(audio)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            for i in range(tonnetz.shape[0]):
                features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
        except Exception:
                features[f'tonnetz_{i}_mean'] = 0.0
        
        # 12. Vocoder Artifact Features (High-Frequency Analysis)
        try:
            # Calculate High-Frequency Energy Ratio (>4kHz vs <4kHz)
            # Neural vocoders often display anomalies in high frequency bands
            S = np.abs(librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH))
            
            # Calculate bin index for 4000Hz cutoff
            # Bin Hz = sr / n_fft. Index = freq / (sr/n_fft) = freq * n_fft / sr
            cutoff_bin = int(4000 * self.N_FFT / sr)
            
            if cutoff_bin < S.shape[0]:
                high_freq_energy = np.sum(S[cutoff_bin:, :])
                low_freq_energy = np.sum(S[:cutoff_bin, :])
                hf_energy_ratio = high_freq_energy / (low_freq_energy + 1e-6)
                features['hf_energy_ratio'] = float(hf_energy_ratio)
            else:
                 features['hf_energy_ratio'] = 0.0
                 
        except Exception:
            features['hf_energy_ratio'] = 0.0
        
        return features
    
    def process_audio(self, audio_base64: str) -> Dict[str, Any]:
        """
        Full pipeline: decode, convert, and extract features.
        
        Args:
            audio_base64: Base64 encoded audio (MP3 or WAV)
            
        Returns:
            Dictionary of extracted features
        """
        # Decode Base64
        audio_bytes = self.decode_base64_audio(audio_base64)
        
        # Convert to samples (auto-detects MP3 or WAV)
        audio, sr = self.convert_audio_to_samples(audio_bytes)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        return features
    
    def process_audio_with_samples(self, audio_base64: str) -> Tuple[Dict[str, Any], np.ndarray, int]:
        """
        Full pipeline: decode, convert, extract features, and return raw audio.
        
        Args:
            audio_base64: Base64 encoded audio (MP3 or WAV)
            
        Returns:
            Tuple of (features dict, audio samples, sample rate)
        """
        # Decode Base64
        audio_bytes = self.decode_base64_audio(audio_base64)
        
        # Convert to samples (auto-detects MP3 or WAV)
        audio, sr = self.convert_audio_to_samples(audio_bytes)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        return features, audio, sr
    
    def generate_spectrogram_base64(self, audio: np.ndarray, sr: int) -> str:
        """Generates a Mel-Spectrogram visualization as Base64 image."""
        try:
            plt.figure(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"Spectrogram error: {e}")
            return None


# Singleton instance
audio_processor = AudioProcessor()
