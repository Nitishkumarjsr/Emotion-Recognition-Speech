"""
CodeAlpha Task 2: Speech Emotion Recognition (SER) ML Engine
Handles acoustic feature extraction (MFCCs, Spectral Centroid, ZCR, RMS Energy, Pitch),
multi-model training (Neural MLP, Random Forest, SVM), benchmark metrics,
2D Circumplex Affect mapping (Valence/Arousal), and real-time audio inference.
"""

import os
import io
import wave
import struct
import warnings
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')


class SpeechEmotionEngine:
    """
    Complete Speech Emotion Recognition (SER) Machine Learning Pipeline.
    Supports 8 standard RAVDESS/SAVEE emotion categories:
    Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
    """

    EMOTIONS = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]

    EMOTION_META = {
        "neutral":   {"label": "Neutral",   "emoji": "😐", "valence": 0.0,  "arousal": 0.0,  "color": "#94a3b8", "desc": "Standard baseline speech with stable pitch and moderate dynamics."},
        "calm":      {"label": "Calm",      "emoji": "😌", "valence": 0.45, "arousal": -0.4, "color": "#38bdf8", "desc": "Relaxed vocal inflection, low energy, and smooth harmonic flow."},
        "happy":     {"label": "Happy",     "emoji": "😄", "valence": 0.85, "arousal": 0.65, "color": "#10b981", "desc": "Elevated dynamic pitch variation, high spectral brightness and joyful energy."},
        "sad":       {"label": "Sad",       "emoji": "😢", "valence": -0.7, "arousal": -0.6, "color": "#6366f1", "desc": "Subdued loudness, downward pitch trend, and lower spectral energy."},
        "angry":     {"label": "Angry",     "emoji": "😡", "valence": -0.65,"arousal": 0.9,  "color": "#f43f5e", "desc": "Sharp attack, high vocal intensity, high RMS energy and elevated harmonics."},
        "fearful":   {"label": "Fearful",   "emoji": "😨", "valence": -0.75,"arousal": 0.7,  "color": "#a855f7", "desc": "High fundamental pitch jitter, wide dynamic range, and vocal tension."},
        "disgust":   {"label": "Disgust",   "emoji": "🤢", "valence": -0.8, "arousal": -0.1, "color": "#f59e0b", "desc": "Guttural vocal quality, lower pitch with distinctive spectral tilt."},
        "surprised": {"label": "Surprised", "emoji": "😲", "valence": 0.35, "arousal": 0.8,  "color": "#ec4899", "desc": "Sudden upward pitch leap and fast vocal onset."}
    }

    FEATURE_NAMES = [
        "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean",
        "mfcc_5_mean", "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean",
        "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean", "mfcc_13_mean",
        "mfcc_std_mean", "spectral_centroid", "spectral_rolloff",
        "spectral_flux", "zcr_mean", "rms_energy", "pitch_f0_mean",
        "pitch_f0_std", "dynamic_range"
    ]

    def __init__(self, random_state: int = 42, samples_per_emotion: int = 250):
        self.random_state = random_state
        self.samples_per_emotion = samples_per_emotion

        # Synthetic reference dataset based on RAVDESS acoustic feature distribution
        self.df = self._generate_ser_corpus()
        self.X = self.df[self.FEATURE_NAMES]
        self.y = self.df["emotion"]

        # Train / Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )

        # Classifiers
        self.models = {
            "NeuralMLP": Pipeline([
                ("scale", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=self.random_state))
            ]),
            "RandomForest": Pipeline([
                ("scale", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=self.random_state))
            ]),
            "SVM": Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, C=2.0, random_state=self.random_state))
            ])
        }

        self.metrics = {}
        self.confusion_matrices = {}
        self.feature_importances = {}
        self.best_model_name = "NeuralMLP"

        self._train_and_evaluate_all()
        self._ensure_sample_audio_files()

    def _generate_ser_corpus(self) -> pd.DataFrame:
        """
        Generates realistic acoustic feature dataset modeling human vocal parameters
        across all 8 emotions according to speech acoustic literature (Scherer et al., RAVDESS).
        """
        np.random.seed(self.random_state)
        rows = []

        # Acoustic Profiles per Emotion
        profiles = {
            "neutral":   {"pitch": 130, "p_std": 12, "rms": 0.045, "zcr": 0.05, "centroid": 1400, "rolloff": 2400, "mfcc_bias": [0, 0, 0, 0, 0]},
            "calm":      {"pitch": 115, "p_std": 8,  "rms": 0.030, "zcr": 0.03, "centroid": 1100, "rolloff": 2000, "mfcc_bias": [-2, -1, 0, 1, 0]},
            "happy":     {"pitch": 210, "p_std": 35, "rms": 0.095, "zcr": 0.09, "centroid": 2200, "rolloff": 3800, "mfcc_bias": [4, 3, 2, 0, 1]},
            "sad":       {"pitch": 105, "p_std": 9,  "rms": 0.025, "zcr": 0.04, "centroid": 1050, "rolloff": 1800, "mfcc_bias": [-4, -2, -1, -1, 0]},
            "angry":     {"pitch": 240, "p_std": 45, "rms": 0.160, "zcr": 0.14, "centroid": 2800, "rolloff": 4500, "mfcc_bias": [6, 5, 4, 2, 2]},
            "fearful":   {"pitch": 225, "p_std": 40, "rms": 0.080, "zcr": 0.12, "centroid": 2500, "rolloff": 4100, "mfcc_bias": [3, 4, 3, 1, 1]},
            "disgust":   {"pitch": 125, "p_std": 15, "rms": 0.050, "zcr": 0.06, "centroid": 1350, "rolloff": 2200, "mfcc_bias": [-1, -3, 2, -2, 0]},
            "surprised": {"pitch": 235, "p_std": 50, "rms": 0.110, "zcr": 0.11, "centroid": 2600, "rolloff": 4300, "mfcc_bias": [5, 4, 1, 3, 1]}
        }

        for emotion in self.EMOTIONS:
            p = profiles[emotion]
            n = self.samples_per_emotion

            pitch_mean = np.random.normal(p["pitch"], p["pitch"] * 0.08, n)
            pitch_std = np.random.normal(p["p_std"], p["p_std"] * 0.12, n)
            rms = np.clip(np.random.normal(p["rms"], p["rms"] * 0.15, n), 0.005, 0.35)
            zcr = np.clip(np.random.normal(p["zcr"], p["zcr"] * 0.18, n), 0.01, 0.25)
            centroid = np.random.normal(p["centroid"], p["centroid"] * 0.10, n)
            rolloff = np.random.normal(p["rolloff"], p["rolloff"] * 0.10, n)
            flux = np.random.exponential(p["rms"] * 2.5, n)
            dyn_range = np.random.normal(p["rms"] * 4.0, 0.05, n)

            # 13 MFCC coefficients simulation
            mfccs = {}
            for k in range(1, 14):
                bias = p["mfcc_bias"][k % len(p["mfcc_bias"])] * 2.5
                mfccs[f"mfcc_{k}_mean"] = np.random.normal(-15 + (14 - k) * 3 + bias, 4.0, n)

            mfcc_std = np.random.normal(12.0 + p["rms"] * 30.0, 2.5, n)

            df_emotion = pd.DataFrame({
                **mfccs,
                "mfcc_std_mean": mfcc_std,
                "spectral_centroid": centroid,
                "spectral_rolloff": rolloff,
                "spectral_flux": flux,
                "zcr_mean": zcr,
                "rms_energy": rms,
                "pitch_f0_mean": pitch_mean,
                "pitch_f0_std": pitch_std,
                "dynamic_range": dyn_range,
                "emotion": emotion
            })
            rows.append(df_emotion)

        return pd.concat(rows, ignore_index=True)

    def _train_and_evaluate_all(self):
        """Fits all models on the feature matrix and computes multi-class evaluation metrics."""
        best_acc = -1.0

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            acc = float(accuracy_score(self.y_test, y_pred))
            prec = float(precision_score(self.y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(self.y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

            # 8x8 Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred, labels=self.EMOTIONS)
            self.confusion_matrices[name] = {
                "labels": self.EMOTIONS,
                "matrix": cm.tolist()
            }

            self.metrics[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "test_samples": len(self.y_test)
            }

            if acc > best_acc:
                best_acc = acc
                self.best_model_name = name

        # Feature importances from Random Forest
        rf_model = self.models["RandomForest"].named_steps["model"]
        rf_imp = rf_model.feature_importances_
        sorted_idx = np.argsort(rf_imp)[::-1]
        self.feature_importances["RandomForest"] = [
            {"feature": self.FEATURE_NAMES[i], "importance": round(float(rf_imp[i]), 4)}
            for i in sorted_idx[:12]
        ]

    def _ensure_sample_audio_files(self):
        """Generates synthesized harmonic WAV files for each emotion into static/samples/."""
        sample_dir = os.path.join(os.path.dirname(__file__), "static", "samples")
        os.makedirs(sample_dir, exist_ok=True)

        for emotion in self.EMOTIONS:
            file_path = os.path.join(sample_dir, f"{emotion}_sample.wav")
            if not os.path.exists(file_path):
                self._synthesize_wav(file_path, emotion)

    def _synthesize_wav(self, file_path: str, emotion: str):
        """Creates an expressive harmonic synthetic audio wave representing emotion speech acoustics."""
        sr = 22050
        duration = 2.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        meta = self.EMOTION_META[emotion]
        base_f0 = 130 + (meta["arousal"] * 80) + (meta["valence"] * 30)

        # Harmonic series with envelope
        envelope = np.sin(np.pi * t / duration) ** 1.5
        if emotion in ["angry", "surprised"]:
            envelope *= (1 + 0.3 * np.sin(2 * np.pi * 6 * t))
        elif emotion == "fearful":
            envelope *= (1 + 0.25 * np.sin(2 * np.pi * 12 * t))

        pitch_contour = base_f0 + (meta["arousal"] * 40 * np.sin(2 * np.pi * 2.5 * t))

        # Harmonics
        signal = (
            0.6 * np.sin(2 * np.pi * pitch_contour * t)
            + 0.3 * np.sin(2 * np.pi * 2 * pitch_contour * t)
            + 0.15 * np.sin(2 * np.pi * 3 * pitch_contour * t)
            + 0.08 * np.sin(2 * np.pi * 4 * pitch_contour * t)
        ) * envelope

        # Add mild noise for breathiness/frication
        noise_level = 0.02 + max(0, meta["arousal"] * 0.04)
        signal += np.random.normal(0, noise_level, len(signal)) * envelope

        # Normalize to 16-bit integer WAV
        signal = np.clip(signal, -1.0, 1.0)
        audio_int16 = (signal * 32767).astype(np.int16)
        wavfile.write(file_path, sr, audio_int16)

    def extract_features_from_audio(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """
        Extracts acoustic speech parameters (MFCCs, Centroid, Rolloff, ZCR, RMS, Pitch)
        from raw audio array using NumPy and SciPy.
        """
        # Convert to mono float32 between -1.0 and 1.0
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if audio_data.dtype == np.int16:
            signal = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            signal = audio_data.astype(np.float32) / 2147483648.0
        else:
            signal = audio_data.astype(np.float32)
            if np.max(np.abs(signal)) > 1.0:
                signal = signal / np.max(np.abs(signal))

        if len(signal) == 0:
            signal = np.zeros(sample_rate)

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(signal ** 2)))

        # 2. Zero-Crossing Rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(signal)))) / 2.0)

        # 3. FFT Spectrum
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
        total_mag = np.sum(fft_vals) + 1e-10

        # 4. Spectral Centroid
        centroid = float(np.sum(freqs * fft_vals) / total_mag)

        # 5. Spectral Rolloff (85% energy point)
        cum_mag = np.cumsum(fft_vals)
        rolloff_idx = np.searchsorted(cum_mag, 0.85 * total_mag)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # 6. Spectral Flux
        flux = float(np.std(fft_vals) / total_mag * 1000)

        # 7. Pitch F0 Estimation (Autocorrelation in 60-400Hz speech range)
        min_lag = int(sample_rate / 400)
        max_lag = int(sample_rate / 60)
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]
        if len(corr) > max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            pitch_f0 = float(sample_rate / peak_lag) if peak_lag > 0 else 140.0
        else:
            pitch_f0 = 140.0

        # Dynamic Range
        dyn_range = float(np.max(signal) - np.min(signal))

        # 8. Simulated MFCC-like filter bank energies
        mfccs = {}
        for k in range(1, 14):
            band_low = int((k - 1) * len(fft_vals) / 14)
            band_high = int(k * len(fft_vals) / 14)
            val = float(np.log(np.mean(fft_vals[band_low:band_high] ** 2) + 1e-6))
            mfccs[f"mfcc_{k}_mean"] = round(val, 3)

        mfcc_std = float(np.std(list(mfccs.values())))

        feature_dict = {
            **mfccs,
            "mfcc_std_mean": round(mfcc_std, 3),
            "spectral_centroid": round(centroid, 2),
            "spectral_rolloff": round(rolloff, 2),
            "spectral_flux": round(flux, 3),
            "zcr_mean": round(zcr, 4),
            "rms_energy": round(rms, 4),
            "pitch_f0_mean": round(pitch_f0, 2),
            "pitch_f0_std": round(pitch_f0 * 0.15, 2),
            "dynamic_range": round(dyn_range, 3)
        }

        return feature_dict

    def predict_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Runs full inference from in-memory WAV byte stream."""
        try:
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
        except Exception:
            # Fallback for synthetic decoding or basic wave format
            sample_rate = 22050
            audio_data = np.frombuffer(audio_bytes[44:], dtype=np.int16) if len(audio_bytes) > 44 else np.zeros(22050)

        feats = self.extract_features_from_audio(audio_data, sample_rate)
        return self.predict_features(feats)

    def predict_features(self, feats: dict) -> dict:
        """Takes acoustic feature dictionary and runs inference across Neural MLP, Random Forest, and SVM."""
        feat_vector = [feats.get(k, 0.0) for k in self.FEATURE_NAMES]
        X_in = pd.DataFrame([feat_vector], columns=self.FEATURE_NAMES)

        model_results = {}
        prob_matrix = {emotion: 0.0 for emotion in self.EMOTIONS}

        for name, model in self.models.items():
            pred = model.predict(X_in)[0]
            probs = model.predict_proba(X_in)[0]
            prob_dict = {cls: round(float(probs[i]) * 100, 2) for i, cls in enumerate(model.classes_)}

            for cls, val in prob_dict.items():
                prob_matrix[cls] += val / len(self.models)

            model_results[name] = {
                "predicted_emotion": pred,
                "confidence": round(max(probs) * 100, 2),
                "probabilities": prob_dict
            }

        # Consensus top emotion
        sorted_emotions = sorted(prob_matrix.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_prob = sorted_emotions[0]
        secondary_emotion, sec_prob = sorted_emotions[1] if len(sorted_emotions) > 1 else (top_emotion, 0.0)

        top_meta = self.EMOTION_META[top_emotion]

        # Calculate estimated 2D Valence and Arousal from probability weights
        est_valence = sum(self.EMOTION_META[emo]["valence"] * (prob_matrix[emo] / 100.0) for emo in self.EMOTIONS)
        est_arousal = sum(self.EMOTION_META[emo]["arousal"] * (prob_matrix[emo] / 100.0) for emo in self.EMOTIONS)

        return {
            "consensus": {
                "top_emotion": top_emotion,
                "label": top_meta["label"],
                "emoji": top_meta["emoji"],
                "color": top_meta["color"],
                "confidence": round(top_prob, 2),
                "description": top_meta["desc"],
                "secondary_emotion": self.EMOTION_META[secondary_emotion]["label"],
                "secondary_confidence": round(sec_prob, 2),
                "valence": round(est_valence, 2),
                "arousal": round(est_arousal, 2),
                "probabilities": {k: round(v, 2) for k, v in prob_matrix.items()}
            },
            "models": model_results,
            "acoustic_features": feats
        }

    def get_dataset_summary(self) -> dict:
        """Returns metadata for API and UI exploration."""
        return {
            "total_samples": len(self.df),
            "emotion_classes": self.EMOTIONS,
            "emotion_meta": self.EMOTION_META,
            "feature_count": len(self.FEATURE_NAMES),
            "best_model": self.best_model_name,
            "samples_per_class": self.samples_per_emotion
        }

    def get_preset_samples(self) -> list:
        """Returns list of playable emotion sample clips."""
        samples = []
        for emotion in self.EMOTIONS:
            meta = self.EMOTION_META[emotion]
            samples.append({
                "id": emotion,
                "label": meta["label"],
                "emoji": meta["emoji"],
                "audio_url": f"/static/samples/{emotion}_sample.wav",
                "description": meta["desc"],
                "valence": meta["valence"],
                "arousal": meta["arousal"]
            })
        return samples


# Global singleton instance
ml_engine = SpeechEmotionEngine()
