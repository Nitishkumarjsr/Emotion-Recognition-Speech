# 🎙️ VoxEmotion AI &mdash; Speech Emotion Recognition & Acoustic Intelligence
**CodeAlpha Machine Learning Internship &bull; Task 2**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Render](https://img.shields.io/badge/backend-Render-46E3B7.svg)](https://render.com)
[![Supabase](https://img.shields.io/badge/database-Supabase-3ECF8E.svg)](https://supabase.com)
[![Vercel](https://img.shields.io/badge/frontend-Vercel-black.svg)](https://vercel.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end cloud-native **Speech Emotion Recognition (SER)** system, acoustic intelligence pipeline, and real-time comparative ML evaluation platform built with:
- **Backend on Render**: Python Flask / Gunicorn REST API with CORS support.
- **Database on Supabase**: PostgreSQL cloud database storing real-time emotion telemetry, biometrics, valence/arousal affect coordinates, and user feedback with Row Level Security (RLS).
- **Frontend on Vercel**: Cyber-Acoustic Glassmorphism Single-Page Application (SPA) with live microphone waveform visualizer, 2D Circumplex plane, live Supabase feed, and dynamic backend switcher.

---

## 🏛️ 3-Tier Cloud Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     VERCEL FRONTEND                      │
│   - Live Microphone Recording & WebAudio Waveform        │
│   - 2D Russell Circumplex Affect Canvas (Valence/Arousal)│
│   - Real-Time Supabase Cloud Prediction Feed & Analytics │
│   - Dynamic Backend Connection Switcher                  │
└────────────────────────────┬─────────────────────────────┘
                             │
                  REST API & Audio Upload
                             ▼
┌──────────────────────────────────────────────────────────┐
│                      RENDER BACKEND                      │
│   - Python Flask & Gunicorn Web Service                  │
│   - Acoustic Feature Extraction (MFCC, F0, Centroid, ZCR)│
│   - Tri-Model Ensemble (Neural MLP, Random Forest, SVM)  │
│   - Health Monitoring & REST Endpoints                   │
└────────────────────────────┬─────────────────────────────┘
                             │
               PostgreSQL / RLS Data Persistence
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    SUPABASE DATABASE                     │
│   - Table: predictions (Biometrics, Affect, Inferences)  │
│   - Table: emotion_feedback (Ratings & Ground Truth)     │
│   - Table: preset_samples (Audio Library Metadata)       │
└──────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

- **Interactive Voice Lab Studio**:
  - **Live Microphone Recording**: Record directly in the browser with real-time waveform visualizer via HTML5 `MediaRecorder` & Web Audio API.
  - **Audio File Upload**: Drag-and-drop `.wav`, `.mp3`, `.ogg`, `.webm` files.
  - **Reference Voice Bank**: 8 instant-play harmonic audio clips for each emotion class.
- **Acoustic Biometric Extraction**: Computes MFCCs (13 coefficients), Pitch ($F_0$), Spectral Centroid, Zero-Crossing Rate (ZCR), and RMS Energy in real time.
- **2D Circumplex Model of Affect**: Visualizes emotional valence (pleasantness) vs arousal (vocal intensity).
- **Supabase Cloud Database Integration**: Real-time logging of prediction events, biometrics, and user feedback ratings with automatic in-memory fallback.
- **Multi-Model Intelligence**: Tri-model consensus across Neural MLP, Random Forest, and Support Vector Machines.
- **Model Evaluation Dashboard**: 8x8 confusion matrix heatmap, classification report metrics, and acoustic feature rankings.
- **Batch Audio Evaluation**: Multi-file batch evaluation with summary statistics.
- **Developer REST API Playground**: Interactive cURL, Python, and JavaScript snippets matching your active backend URL.

---

## 📁 Repository Structure

```
CodeAlpha_EmotionRecognitionSpeech/
├── render.yaml                         # Render Blueprint 1-click deployment
├── Procfile                            # Render / PaaS start command
├── Dockerfile                          # Docker container configuration
├── requirements.txt                    # Python package dependencies
├── .env.example                        # Environment variables template
├── supabase_schema.sql                 # Supabase PostgreSQL schema & migration script
├── DEPLOYMENT_GUIDE.md                 # Step-by-step Render, Supabase & Vercel guide
│
├── app.py                              # Flask REST API server (Render Backend)
├── supabase_client.py                  # Supabase database manager & cloud layer
├── ml_engine.py                        # Acoustic extraction & SER ML engine
├── run_web_app.py                      # One-click local browser launcher
├── speech_emotion.py                   # Standalone CLI training & evaluation script
├── speech_emotion_model.joblib         # Serialized top-performing ML pipeline
├── model_results.csv                   # Baseline benchmark metrics
│
├── frontend/                           # Standalone Frontend (Vercel Ready)
│   ├── index.html                      # Vercel static SPA dashboard
│   ├── vercel.json                     # Vercel deployment configuration
│   ├── css/
│   │   └── style.css                   # Cyber-Acoustic glassmorphism styling
│   ├── js/
│   │   └── app.js                      # Frontend recording, visualizer & API controller
│   └── samples/                        # Reference emotion WAV audio bank
│
├── vercel.json                         # Root Vercel deployment redirect
├── templates/
│   └── index.html                      # Local Flask server dashboard
├── static/
│   ├── css/
│   │   └── style.css                   # Flask static CSS
│   ├── js/
│   │   └── app.js                      # Flask static JS
│   └── samples/                        # Flask static audio samples
│
├── WEB_APP_GUIDE.md                    # Comprehensive web application guide
└── README.md                           # Project documentation
```

---

## 🚀 Quick Deployment Guide

See **[DEPLOYMENT_GUIDE.md](file:///c:/Users/lenovo/Downloads/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_EmotionRecognitionSpeech/DEPLOYMENT_GUIDE.md)** for detailed instructions.

### 1. Supabase (Database)
1. Create a project at [supabase.com](https://supabase.com).
2. Open **SQL Editor**, paste and execute [`supabase_schema.sql`](file:///c:/Users/lenovo/Downloads/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_EmotionRecognitionSpeech/supabase_schema.sql).
3. Copy your `Project URL` and `anon key` from Project Settings &rarr; API.

### 2. Render (Backend)
1. Create a new **Web Service** on [dashboard.render.com](https://dashboard.render.com) connected to this repository.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Set Environment Variables:
   - `SUPABASE_URL`: `https://your-project.supabase.co`
   - `SUPABASE_KEY`: `your-supabase-anon-or-service-key`
   - `CORS_ORIGINS`: `*`

### 3. Vercel (Frontend)
1. Import repository on [vercel.com](https://vercel.com).
2. Set Root Directory to `frontend` (or keep `./` with included `vercel.json`).
3. Deploy!
4. Open the deployed Vercel site, click **⚙️ API Config**, and enter your Render backend URL.

---

## 💻 Local Development

```bash
# 1. Clone repository
git clone <your-github-repo-url>
cd CodeAlpha_EmotionRecognitionSpeech

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask Server
python app.py
```
Open **[http://127.0.0.1:5002](http://127.0.0.1:5002)** in your browser.

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
