# VoxEmotion AI &mdash; Web Application Guide
**Comprehensive Architecture, Acoustic Feature Pipeline, and REST API Documentation**

---

## 1. System Overview
**VoxEmotion AI** is a real-time Speech Emotion Recognition (SER) web application built with Scikit-Learn, SciPy, and Flask. It analyzes vocal audio, extracts 22 acoustic spectral features, classifies 8 primary human emotions, and computes dimensional affective coordinates (*Valence vs Arousal*).

---

## 2. Acoustic Feature Extraction Pipeline

The feature extraction engine operates on raw audio streams (sampled at 22,050 Hz):
1. **MFCCs (13 Coefficients)**: Captures the spectral envelope and vocal tract formant resonances.
2. **Spectral Centroid**: The center of mass of the frequency spectrum, indicating vocal sharpness / brightness.
3. **Spectral Rolloff**: The 85% energy frequency threshold.
4. **Spectral Flux**: Rate of spectral energy variation over time.
5. **Zero-Crossing Rate (ZCR)**: Rate at which the audio signal changes sign (differentiates unvoiced/fricative noise from harmonic voice).
6. **RMS Energy**: Root Mean Square energy representing vocal loudness and dynamic intensity.
7. **Fundamental Frequency (Pitch / F0)**: Autocorrelation-based pitch tracking in human speech range (60&ndash;400 Hz).
8. **2D Circumplex Mapping**: Computes Russell's emotional dimensions:
   - **Valence**: Pleasantness vs unpleasantness (-1.0 to +1.0).
   - **Arousal**: Physiological activation and vocal energy (-1.0 to +1.0).

---

## 3. REST API Reference

### `GET /api/summary`
Returns emotion categories, metadata, and feature definitions.

### `GET /api/metrics`
Returns multi-model performance benchmarks, 8x8 confusion matrices, and feature importances.

### `GET /api/samples`
Returns all 8 pre-loaded emotion voice clips with playable links.

### `GET /api/sample/<emotion_id>`
Executes inference on a specific reference voice sample.

### `POST /api/predict`
Accepts multipart audio upload, base64 audio string, or acoustic feature vector.

**Request (Multipart File Upload):**
```bash
curl -X POST http://127.0.0.1:5002/api/predict \
  -F "file=@speech_clip.wav"
```

**Response Payload:**
```json
{
  "status": "success",
  "result": {
    "consensus": {
      "top_emotion": "happy",
      "label": "Happy",
      "emoji": "😄",
      "confidence": 94.8,
      "valence": 0.85,
      "arousal": 0.65,
      "secondary_emotion": "Surprised",
      "secondary_confidence": 4.2,
      "probabilities": {
        "happy": 94.8,
        "surprised": 4.2,
        "calm": 0.6,
        "neutral": 0.2,
        "angry": 0.1,
        "fearful": 0.1,
        "sad": 0.0,
        "disgust": 0.0
      }
    },
    "models": { ... },
    "acoustic_features": {
      "pitch_f0_mean": 210.0,
      "rms_energy": 0.095,
      "spectral_centroid": 2200.0,
      "zcr_mean": 0.090
    }
  }
}
```
