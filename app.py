"""
CodeAlpha Task 2: VoxEmotion AI - Speech Emotion Recognition Web App
Flask Application & Secure Audio REST API Server
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from ml_engine import ml_engine

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB audio file upload limit


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
    """Main Web Application Dashboard."""
    summary = ml_engine.get_dataset_summary()
    samples = ml_engine.get_preset_samples()
    return render_template("index.html", summary=summary, samples=samples)


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
    """Returns multi-model performance benchmarks, 8x8 confusion matrices, and acoustic feature rankings."""
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
            return jsonify({"status": "error", "message": "Sample not found."}), 404

        with open(sample_path, "rb") as f:
            audio_bytes = f.read()

        result = ml_engine.predict_audio_bytes(audio_bytes, filename=f"{emotion_id}.wav")
        return jsonify({"status": "success", "result": result, "audio_url": f"/static/samples/{emotion_id}_sample.wav"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts an uploaded audio file (multipart/form-data), base64 encoded audio string,
    or acoustic feature JSON payload, and returns real-time multi-model emotion probabilities.
    """
    try:
        # 1. Check for file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No audio file selected."}), 400
            
            audio_bytes = file.read()
            result = ml_engine.predict_audio_bytes(audio_bytes, filename=file.filename)
            return jsonify({"status": "success", "result": result})

        # 2. Check for JSON / base64 payload
        req_data = request.get_json(force=True, silent=True) or {}
        
        if "audio_base64" in req_data:
            base64_str = req_data["audio_base64"]
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            audio_bytes = base64.b64decode(base64_str)
            result = ml_engine.predict_audio_bytes(audio_bytes, filename="recording.wav")
            return jsonify({"status": "success", "result": result})

        if "features" in req_data:
            result = ml_engine.predict_features(req_data["features"])
            return jsonify({"status": "success", "result": result})

        return jsonify({"status": "error", "message": "No valid audio data or features supplied."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Processes batch evaluation of audio files or feature vectors."""
    try:
        results = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            for i, f in enumerate(files[:50]):
                audio_bytes = f.read()
                res = ml_engine.predict_audio_bytes(audio_bytes, filename=f.filename)
                results.append({
                    "id": i + 1,
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
                    res = ml_engine.predict_audio_bytes(audio_bytes, filename=f"{emotion}.wav")
                    results.append({
                        "id": len(results) + 1,
                        "filename": f"{emotion}_sample.wav",
                        "top_emotion": res["consensus"]["label"],
                        "emoji": res["consensus"]["emoji"],
                        "confidence": res["consensus"]["confidence"],
                        "valence": res["consensus"]["valence"],
                        "arousal": res["consensus"]["arousal"]
                    })

        return jsonify({
            "status": "success",
            "total_processed": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("\n========================================================")
    print("VoxEmotion AI: Speech Emotion Recognition Web App Running!")
    print(f"Access Dashboard at: http://127.0.0.1:{port}")
    print("========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
