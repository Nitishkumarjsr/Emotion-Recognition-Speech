"""
CodeAlpha Task 2: VoxEmotion AI - Speech Emotion Recognition Web App
Production REST API Server with Render Backend Support, Supabase PostgreSQL Integration & CORS for Vercel
"""

import os
import io
import base64
import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from ml_engine import ml_engine
from supabase_client import db_manager

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB audio upload limit

# Configure CORS to allow Vercel frontend domains and local dev
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
else:
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)


@app.after_request
def set_security_headers(response):
    """Adds standard security headers to all HTTP responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route("/")
def index():
    """Web Application Dashboard (local or server preview)."""
    summary = ml_engine.get_dataset_summary()
    samples = ml_engine.get_preset_samples()
    supabase_status = db_manager.check_health()
    return render_template("index.html", summary=summary, samples=samples, supabase_status=supabase_status)


# ============================================================================
# HEALTH & SYSTEM STATUS (FOR RENDER & VERCEL MONITORING)
# ============================================================================
@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Health check endpoint for Render, uptime monitors, and frontend status checks.
    Reports ML pipeline readiness, active models, and Supabase connection state.
    """
    try:
        supabase_health = db_manager.check_health()
        return jsonify({
            "status": "online",
            "service": "VoxEmotion AI REST API",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ml_engine": {
                "status": "ready",
                "top_model": ml_engine.best_model_name,
                "supported_emotions": ml_engine.EMOTIONS,
                "feature_count": len(ml_engine.FEATURE_NAMES)
            },
            "database": supabase_health
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# ML DATASET & BENCHMARK METRICS
# ============================================================================
@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Returns emotion metadata, feature names, and class distributions."""
    try:
        data = ml_engine.get_dataset_summary()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns multi-model performance benchmarks, confusion matrices, and acoustic rankings."""
    try:
        return jsonify({
            "status": "success",
            "metrics": ml_engine.metrics,
            "confusion_matrices": ml_engine.confusion_matrices,
            "feature_importances": ml_engine.feature_importances,
            "best_model": ml_engine.best_model_name,
            "emotion_meta": ml_engine.EMOTION_META
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# REFERENCE AUDIO SAMPLES
# ============================================================================
@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns all pre-loaded emotion voice clips."""
    try:
        samples = ml_engine.get_preset_samples()
        return jsonify({"status": "success", "samples": samples})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample/<emotion_id>", methods=["GET"])
def api_sample_detail(emotion_id):
    """Loads and predicts a preset sample by emotion ID."""
    try:
        sample_path = os.path.join(app.root_path, "static", "samples", f"{emotion_id}_sample.wav")
        if not os.path.exists(sample_path):
            return jsonify({"status": "error", "message": f"Sample for '{emotion_id}' not found."}), 404

        with open(sample_path, "rb") as f:
            audio_bytes = f.read()

        filename = f"{emotion_id}_sample.wav"
        result = ml_engine.predict_audio_bytes(audio_bytes, filename=filename)
        
        # Save prediction log to Supabase
        db_record = db_manager.save_prediction(result, filename=filename, session_id="sample_preview")

        return jsonify({
            "status": "success",
            "result": result,
            "db_id": db_record.get("id"),
            "audio_url": f"/static/samples/{emotion_id}_sample.wav"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# REAL-TIME EMOTION PREDICTION (FILE / BASE64 / FEATURES)
# ============================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts an uploaded audio file (multipart/form-data), base64 encoded audio string,
    or acoustic feature JSON payload, computes multi-model emotion probabilities,
    and automatically persists the prediction to Supabase database.
    """
    try:
        filename = "voice_recording.wav"
        session_id = request.headers.get("X-Session-ID", "web_client")
        result = None

        # 1. Check for multipart file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No audio file selected."}), 400
            
            filename = file.filename
            audio_bytes = file.read()
            result = ml_engine.predict_audio_bytes(audio_bytes, filename=filename)

        # 2. Check for JSON / base64 payload
        else:
            req_data = request.get_json(force=True, silent=True) or {}
            
            if "audio_base64" in req_data:
                base64_str = req_data["audio_base64"]
                filename = req_data.get("filename", "microphone_stream.wav")
                if "," in base64_str:
                    base64_str = base64_str.split(",")[1]
                audio_bytes = base64.b64decode(base64_str)
                result = ml_engine.predict_audio_bytes(audio_bytes, filename=filename)

            elif "features" in req_data:
                filename = req_data.get("filename", "custom_features.json")
                result = ml_engine.predict_features(req_data["features"])

        if result is None:
            return jsonify({"status": "error", "message": "No valid audio data or features supplied."}), 400

        # Persist prediction event to Supabase
        db_record = db_manager.save_prediction(result, filename=filename, session_id=session_id)

        return jsonify({
            "status": "success",
            "result": result,
            "prediction_id": db_record.get("id"),
            "created_at": db_record.get("created_at"),
            "supabase_synced": db_manager.is_configured()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# BATCH PREDICTION
# ============================================================================
@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Processes batch evaluation of multiple audio files."""
    try:
        results = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            for i, f in enumerate(files[:50]):
                audio_bytes = f.read()
                res = ml_engine.predict_audio_bytes(audio_bytes, filename=f.filename)
                db_record = db_manager.save_prediction(res, filename=f.filename, session_id="batch_eval")
                results.append({
                    "id": i + 1,
                    "db_id": db_record.get("id"),
                    "filename": f.filename,
                    "top_emotion": res["consensus"]["label"],
                    "emoji": res["consensus"]["emoji"],
                    "confidence": res["consensus"]["confidence"],
                    "valence": res["consensus"]["valence"],
                    "arousal": res["consensus"]["arousal"]
                })
        else:
            # Generate demonstration batch from all 8 presets
            for emotion in ml_engine.EMOTIONS:
                sample_path = os.path.join(app.root_path, "static", "samples", f"{emotion}_sample.wav")
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        audio_bytes = f.read()
                    fname = f"{emotion}_sample.wav"
                    res = ml_engine.predict_audio_bytes(audio_bytes, filename=fname)
                    db_record = db_manager.save_prediction(res, filename=fname, session_id="demo_batch")
                    results.append({
                        "id": len(results) + 1,
                        "db_id": db_record.get("id"),
                        "filename": fname,
                        "top_emotion": res["consensus"]["label"],
                        "emoji": res["consensus"]["emoji"],
                        "confidence": res["consensus"]["confidence"],
                        "valence": res["consensus"]["valence"],
                        "arousal": res["consensus"]["arousal"]
                    })

        return jsonify({
            "status": "success",
            "total_processed": len(results),
            "results": results,
            "supabase_synced": db_manager.is_configured()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# SUPABASE DATABASE CLOUD HISTORY & ANALYTICS
# ============================================================================
@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Fetches real-time prediction history directly from Supabase PostgreSQL database.
    Query param ?limit=N (default 20, max 100).
    """
    try:
        limit = min(100, int(request.args.get("limit", 20)))
        history = db_manager.get_recent_predictions(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(history),
            "database_status": db_manager.check_health(),
            "history": history
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    Submits user feedback and accuracy ratings into Supabase emotion_feedback table.
    """
    try:
        req_data = request.get_json(force=True, silent=True) or {}
        prediction_id = req_data.get("prediction_id")
        rating = req_data.get("rating", 5)
        user_feedback = req_data.get("user_feedback", "")
        user_corrected_emotion = req_data.get("user_corrected_emotion", "")

        if not prediction_id:
            return jsonify({"status": "error", "message": "prediction_id is required."}), 400

        entry = db_manager.save_feedback(
            prediction_id=prediction_id,
            rating=rating,
            user_feedback=user_feedback,
            user_corrected_emotion=user_corrected_emotion
        )

        return jsonify({
            "status": "success",
            "message": "Feedback recorded successfully in Supabase.",
            "feedback": entry
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """
    Returns aggregated speech emotion analytics from Supabase.
    """
    try:
        analytics = db_manager.get_analytics_summary()
        return jsonify({
            "status": "success",
            "analytics": analytics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("\n" + "=" * 60)
    print("🎙️ VoxEmotion AI - Speech Emotion Recognition REST API")
    print(f"Backend Server: http://127.0.0.1:{port}")
    print("Supabase Status:", "Configured" if db_manager.is_configured() else "Standby (Memory Cache)")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
