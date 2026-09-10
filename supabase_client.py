"""
VoxEmotion AI - Supabase Database Client & Cloud Data Layer
Handles persistent storage of speech emotion inferences, acoustic biometrics,
affective valence/arousal coordinates, and user feedback in Supabase PostgreSQL.
Provides automatic in-memory fallback for local development or unconfigured deployments.
"""

import os
import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

logger = logging.getLogger("VoxEmotion.Supabase")
logging.basicConfig(level=logging.INFO)

# Supabase Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_ANON_KEY", ""))).strip()

# Initialize Client if credentials exist
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        supabase_client: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Supabase client initialized successfully with URL: {SUPABASE_URL}")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase Python client ({e}). Falling back to HTTP/In-memory.")
        supabase_client = None


class SupabaseDataManager:
    """
    Manages all database interactions with Supabase PostgreSQL,
    including prediction history, analytics summaries, and user feedback.
    """

    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.client = supabase_client
        # Local in-memory ring buffer for fallback when Supabase is not yet configured
        self._local_history: List[Dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "filename": "happy_sample.wav",
                "predicted_emotion": "happy",
                "confidence": 94.80,
                "valence": 0.85,
                "arousal": 0.65,
                "secondary_emotion": "surprised",
                "secondary_confidence": 4.20,
                "acoustic_features": {"pitch_f0_mean": 210.0, "rms_energy": 0.095, "spectral_centroid": 2200.0, "zcr_mean": 0.09},
                "model_probabilities": {"happy": 94.8, "surprised": 4.2, "calm": 1.0}
            },
            {
                "id": str(uuid.uuid4()),
                "created_at": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)).isoformat(),
                "filename": "angry_sample.wav",
                "predicted_emotion": "angry",
                "confidence": 96.20,
                "valence": -0.65,
                "arousal": 0.90,
                "secondary_emotion": "fearful",
                "secondary_confidence": 2.80,
                "acoustic_features": {"pitch_f0_mean": 240.0, "rms_energy": 0.160, "spectral_centroid": 2800.0, "zcr_mean": 0.14},
                "model_probabilities": {"angry": 96.2, "fearful": 2.8, "disgust": 1.0}
            },
            {
                "id": str(uuid.uuid4()),
                "created_at": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)).isoformat(),
                "filename": "calm_sample.wav",
                "predicted_emotion": "calm",
                "confidence": 91.50,
                "valence": 0.45,
                "arousal": -0.40,
                "secondary_emotion": "neutral",
                "secondary_confidence": 6.50,
                "acoustic_features": {"pitch_f0_mean": 115.0, "rms_energy": 0.030, "spectral_centroid": 1100.0, "zcr_mean": 0.03},
                "model_probabilities": {"calm": 91.5, "neutral": 6.5, "sad": 2.0}
            }
        ]
        self._local_feedbacks: List[Dict[str, Any]] = []

    def is_configured(self) -> bool:
        """Returns True if Supabase credentials are set."""
        return bool(self.url and self.key and self.client)

    def check_health(self) -> Dict[str, Any]:
        """Tests live connection to Supabase and returns diagnostic status."""
        if not (self.url and self.key):
            return {
                "configured": False,
                "connected": False,
                "status": "standby_local_mode",
                "message": "Supabase credentials not configured in environment (SUPABASE_URL / SUPABASE_KEY). Using memory cache.",
                "database_engine": "In-Memory Fallback"
            }

        if not self.client:
            return {
                "configured": True,
                "connected": False,
                "status": "client_error",
                "message": "Supabase client failed to initialize. Check package installation and URL.",
                "database_engine": "Supabase Client Error"
            }

        try:
            # Query 1 record from predictions table
            res = self.client.table("predictions").select("id").limit(1).execute()
            return {
                "configured": True,
                "connected": True,
                "status": "connected",
                "message": "Successfully connected to Supabase PostgreSQL database!",
                "database_engine": "Supabase PostgreSQL",
                "url": self.url.split("//")[-1].split(".")[0] + ".supabase.co"
            }
        except Exception as e:
            return {
                "configured": True,
                "connected": False,
                "status": "connection_failed",
                "message": f"Connection to Supabase failed: {str(e)}",
                "database_engine": "Supabase (Unreachable)"
            }

    def save_prediction(self, prediction_result: Dict[str, Any], filename: str = "recording.wav", session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Persists speech emotion inference output to Supabase PostgreSQL.
        """
        consensus = prediction_result.get("consensus", {})
        models = prediction_result.get("models", {})
        feats = prediction_result.get("acoustic_features", {})

        record = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "filename": filename,
            "predicted_emotion": consensus.get("top_emotion", "neutral"),
            "confidence": float(consensus.get("confidence", 0.0)),
            "valence": float(consensus.get("valence", 0.0)),
            "arousal": float(consensus.get("arousal", 0.0)),
            "secondary_emotion": consensus.get("secondary_emotion", ""),
            "secondary_confidence": float(consensus.get("secondary_confidence", 0.0)),
            "acoustic_features": feats,
            "model_probabilities": consensus.get("probabilities", {}),
            "models_consensus": models,
            "session_id": session_id or "web_session"
        }

        # Always update local buffer
        self._local_history.insert(0, record)
        if len(self._local_history) > 100:
            self._local_history.pop()

        # If Supabase client is live, push to cloud
        if self.is_configured():
            try:
                self.client.table("predictions").insert(record).execute()
                logger.info(f"Saved prediction {record['id']} to Supabase table 'predictions'.")
            except Exception as e:
                logger.warning(f"Could not write prediction to Supabase ({e}). Stored in local buffer.")

        return record

    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the latest prediction records.
        """
        if self.is_configured():
            try:
                res = self.client.table("predictions") \
                    .select("*") \
                    .order("created_at", desc=True) \
                    .limit(limit) \
                    .execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception as e:
                logger.warning(f"Failed to fetch from Supabase ({e}), returning local history.")

        return self._local_history[:limit]

    def save_feedback(self, prediction_id: str, rating: int, user_feedback: Optional[str] = None, user_corrected_emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Saves user feedback and validation rating (1 to 5 stars) for an emotion prediction.
        """
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "prediction_id": prediction_id,
            "rating": max(1, min(5, int(rating))),
            "user_feedback": (user_feedback or "").strip(),
            "user_corrected_emotion": (user_corrected_emotion or "").strip(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        self._local_feedbacks.insert(0, feedback_entry)

        if self.is_configured():
            try:
                self.client.table("emotion_feedback").insert(feedback_entry).execute()
                logger.info(f"Feedback saved to Supabase for prediction {prediction_id}")
            except Exception as e:
                logger.warning(f"Could not write feedback to Supabase: {e}")

        return feedback_entry

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Aggregates emotion distributions, average confidence, and average affect coordinates.
        """
        predictions = self.get_recent_predictions(limit=100)
        
        total_inferences = len(predictions)
        emotion_counts: Dict[str, int] = {}
        total_conf = 0.0
        total_val = 0.0
        total_aro = 0.0

        for p in predictions:
            emo = p.get("predicted_emotion", "neutral")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            total_conf += float(p.get("confidence", 0.0))
            total_val += float(p.get("valence", 0.0))
            total_aro += float(p.get("arousal", 0.0))

        avg_conf = round(total_conf / max(1, total_inferences), 2)
        avg_val = round(total_val / max(1, total_inferences), 2)
        avg_aro = round(total_aro / max(1, total_inferences), 2)

        return {
            "total_inferences": total_inferences,
            "average_confidence": avg_conf,
            "average_valence": avg_val,
            "average_arousal": avg_aro,
            "emotion_counts": emotion_counts,
            "supabase_status": self.check_health()
        }


# Global Singleton Database Manager
db_manager = SupabaseDataManager()
