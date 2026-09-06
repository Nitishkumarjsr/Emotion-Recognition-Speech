# 🎙️ VoxEmotion AI &mdash; Speech Emotion Recognition & Acoustic Intelligence
**CodeAlpha Machine Learning Internship &bull; Task 2**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Speech Emotion Recognition (SER) web application, acoustic intelligence pipeline, and comparative model evaluation suite.

The system classifies 8 standard human emotions (*Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised*) from audio streams using **Deep Neural MLP**, **Random Forest**, and **Support Vector Machine (RBF)**, mapping vocal acoustics to Russell's 2D Circumplex Model of Affect (*Valence vs Arousal*).

---

## 🌟 Web Application Features

- **Interactive Voice Lab Studio**:
  - **Live Microphone Recording**: Record voice directly in browser with real-time waveform visualizer via HTML5 `MediaRecorder` & Web Audio API.
  - **Audio File Upload**: Drag-and-drop `.wav`, `.mp3`, `.ogg`, `.webm` files.
  - **Reference Voice Bank**: 8 instant-play harmonic audio clips for each emotion class.
- **Acoustic Biometric Extraction**: Computes MFCCs (13 coefficients), Pitch (F0), Spectral Centroid, Zero-Crossing Rate (ZCR), and RMS Energy in real time.
- **2D Circumplex Model of Affect**: Visualizes emotional valence (positivity/negativity) vs arousal (vocal energy).
- **Multi-Model Intelligence**: Tri-model consensus across Neural MLP, Random Forest, and SVM.
- **Model Evaluation Dashboard**: 8x8 confusion matrix heatmap, classification report metrics, and acoustic feature rankings.
- **Speech Audio Library**: Interactive browser gallery with in-app audio players and instant studio loader.
- **Batch Audio Evaluation**: Multi-file batch evaluation with summary statistics.
- **Developer REST API Playground**: Interactive cURL, Python, and JavaScript snippets.

---

## 📁 Repository Structure

```
CodeAlpha_EmotionRecognitionSpeech/
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python package dependencies
├── app.py                              # Flask web server & secure REST API
├── ml_engine.py                        # Acoustic extraction & SER ML engine
├── run_web_app.py                      # One-click browser launcher
├── speech_emotion.py                   # Standalone CLI training & evaluation script
├── speech_emotion_model.joblib         # Serialized top-performing ML pipeline
├── model_results.csv                   # Baseline benchmark metrics
├── templates/
│   └── index.html                      # Dashboard user interface
├── static/
│   ├── css/
│   │   └── style.css                   # Cyber-Acoustic glassmorphism styling
│   ├── js/
│   │   └── app.js                      # Frontend recording, visualizer & charts
│   └── samples/                        # Reference emotion WAV audio bank
├── WEB_APP_GUIDE.md                    # Comprehensive web application guide
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
git clone <your-github-repo-url>
cd CodeAlpha_EmotionRecognitionSpeech
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application

**Option A &mdash; One-Click Launch (Auto-opens browser):**
```bash
python run_web_app.py
```

**Option B &mdash; Direct Flask Server:**
```bash
python app.py
```
Open **[http://127.0.0.1:5002](http://127.0.0.1:5002)** in your browser.

---

## 💻 Standalone CLI Script

To run standalone model training, evaluation, and benchmark generation:
```bash
python speech_emotion.py
```

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Neural MLP (128x64)** | **92.50%** | **0.9265** | **0.9250** | **0.9248** |
| **Random Forest Ensemble (150 Trees)** | **91.00%** | **0.9120** | **0.9100** | **0.9095** |
| **Support Vector Machine (RBF C=2.0)** | **90.50%** | **0.9075** | **0.9050** | **0.9045** |

---

## 📜 License
This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
