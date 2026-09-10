# 🚀 VoxEmotion AI: Production Deployment Guide
## Render (Backend) &bull; Supabase (Database) &bull; Vercel (Frontend)

This guide walks you through deploying **VoxEmotion AI** to a production cloud environment in 3 simple steps:

```
┌─────────────────────────┐       REST API / Audio       ┌─────────────────────────┐
│     Vercel Frontend     │ ───────────────────────────> │      Render Backend     │
│   (HTML5 / CSS3 / JS)   │                              │     (Python / Flask)    │
└─────────────────────────┘                              └────────────┬────────────┘
             │                                                        │
             │           Real-Time Cloud Persistence                  │
             └────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Supabase Database    │
                         │    (PostgreSQL & RLS)   │
                         └─────────────────────────┘
```

---

## 📋 Table of Contents
1. [Step 1: Set Up Supabase Database](#step-1-set-up-supabase-database)
2. [Step 2: Deploy Backend to Render](#step-2-deploy-backend-to-render)
3. [Step 3: Deploy Frontend to Vercel](#step-3-deploy-frontend-to-vercel)
4. [Step 4: Connect & Verify Live System](#step-4-connect--verify-live-system)
5. [Local Development Testing](#local-development-testing)

---

## Step 1: Set Up Supabase Database

1. Go to **[https://supabase.com](https://supabase.com)** and sign in / create a free account.
2. Click **"New Project"**, choose a name (e.g. `voxemotion-db`), set a database password, and choose a region close to your users.
3. Once the project is provisioned (approx 1 minute):
   - Open **SQL Editor** from the left navigation menu.
   - Click **"New Query"**.
   - Open the file [`supabase_schema.sql`](file:///c:/Users/lenovo/Downloads/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_ML_Internship_All_Tasks/CodeAlpha_EmotionRecognitionSpeech/supabase_schema.sql) from this repository, copy the entire SQL script, paste it into the editor, and click **Run**.
4. Retrieve your API credentials:
   - In Supabase, go to **Project Settings (⚙️)** &rarr; **API**.
   - Copy **Project URL** (e.g., `https://xyzprojectid.supabase.co`).
   - Copy **Project API keys** &rarr; `anon` `public` key (or `service_role` key).

> [!NOTE]
> The database schema includes `predictions`, `emotion_feedback`, `preset_samples` tables, indexes, and Row Level Security (RLS) policies.

---

## Step 2: Deploy Backend to Render

1. Go to **[https://dashboard.render.com](https://dashboard.render.com)** and sign in.
2. Push your project code to a GitHub or GitLab repository.
3. Click **"New +"** &rarr; **"Web Service"**.
4. Connect your GitHub repository.
5. Configure the service settings:

| Setting | Value |
| :--- | :--- |
| **Name** | `voxemotion-backend` |
| **Language** | `Python` |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Instance Type** | `Free` or `Starter` |

6. Add the following **Environment Variables** in the Render dashboard:

| Key | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.8` | Recommended Python runtime |
| `CORS_ORIGINS` | `*` | Or specify your Vercel URL (e.g. `https://voxemotion.vercel.app`) |
| `SUPABASE_URL` | `https://your-project.supabase.co` | From Supabase Step 1 |
| `SUPABASE_KEY` | `your-supabase-anon-or-service-key` | From Supabase Step 1 |

7. Click **"Create Web Service"**.
8. Once deployment finishes, copy your Render service URL (e.g., `https://voxemotion-backend.onrender.com`).
9. Verify health in your browser:
   `https://voxemotion-backend.onrender.com/api/health`

---

## Step 3: Deploy Frontend to Vercel

### Option A &mdash; Via Vercel Web Dashboard (Recommended)
1. Go to **[https://vercel.com](https://vercel.com)** and sign in.
2. Click **"Add New..."** &rarr; **"Project"** and import your GitHub repository.
3. In **Project Settings**:
   - **Framework Preset**: `Other`
   - **Root Directory**: Select `frontend` (or leave as root `./` with the included `vercel.json`).
4. Click **"Deploy"**.
5. Once deployed, Vercel gives you an instant domain (e.g., `https://voxemotion.vercel.app`).

### Option B &mdash; Via Vercel CLI
```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend directory & deploy
cd frontend
vercel --prod
```

---

## Step 4: Connect & Verify Live System

1. Open your deployed Vercel frontend URL in any browser.
2. Click the **"⚙️ API Config"** button in the top right navbar.
3. Enter your Render backend URL (e.g., `https://voxemotion-backend.onrender.com`).
4. Click **"Test Connection"** &rarr; You will see a green confirmation checkmark with measured latency.
5. Click **"Save & Connect"**.
6. Test features:
   - 🎙️ **Live Microphone**: Speak into the microphone and check real-time classification.
   - 🎵 **Reference Audio Bank**: Click any emotion preset to listen and classify.
   - ⚡ **Supabase Live Feed**: Scroll to the "Cloud DB" section to see the prediction logged in Supabase with timestamp and valence/arousal affect coordinates.
   - ★ **Rate Prediction**: Click "Rate" on any record to save 1-5 star user feedback to Supabase.

---

## Local Development Testing

To run the full stack locally on your computer:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Copy .env.example to .env and set your Supabase keys
cp .env.example .env

# 3. Start Flask Backend
python app.py
```

Backend will run on **http://127.0.0.1:5002**.

To preview the frontend locally:
- Simply open `http://127.0.0.1:5002` in your browser, OR
- Serve the `frontend/` folder via any static server (e.g. `python -m http.server 3000 --directory frontend` or Live Server).

---

## 🔒 Security & CORS Notes
- CORS is enabled in `app.py` for `/api/*` endpoints.
- If you want to restrict requests to only your Vercel domain on Render, set `CORS_ORIGINS=https://your-app.vercel.app` in Render environment settings.
- Supabase Row Level Security (RLS) is pre-configured to allow public anonymous read and insert for the interactive demo.
