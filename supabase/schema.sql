-- ============================================================================
-- VoxEmotion AI: Speech Emotion Recognition & Acoustic Intelligence
-- Supabase Database Schema & Migration Script
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    filename TEXT DEFAULT 'voice_recording.wav',
    predicted_emotion TEXT NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL,
    valence NUMERIC(4, 2) NOT NULL,
    arousal NUMERIC(4, 2) NOT NULL,
    secondary_emotion TEXT,
    secondary_confidence NUMERIC(5, 2),
    acoustic_features JSONB DEFAULT '{}'::jsonb,
    model_probabilities JSONB DEFAULT '{}'::jsonb,
    models_consensus JSONB DEFAULT '{}'::jsonb,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS public.emotion_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES public.predictions(id) ON DELETE CASCADE,
    user_feedback TEXT,
    user_corrected_emotion TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS public.preset_samples (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    emoji TEXT NOT NULL,
    audio_url TEXT NOT NULL,
    description TEXT,
    valence NUMERIC(4, 2) NOT NULL,
    arousal NUMERIC(4, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_emotion ON public.predictions (predicted_emotion);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON public.predictions (confidence DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id ON public.emotion_feedback (prediction_id);

ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emotion_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.preset_samples ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to predictions" ON public.predictions FOR SELECT USING (true);
CREATE POLICY "Allow public insert to predictions" ON public.predictions FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public read feedback" ON public.emotion_feedback FOR SELECT USING (true);
CREATE POLICY "Allow public insert feedback" ON public.emotion_feedback FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public read presets" ON public.preset_samples FOR SELECT USING (true);

INSERT INTO public.preset_samples (id, label, emoji, audio_url, description, valence, arousal)
VALUES
    ('neutral',   'Neutral',   '😐', '/static/samples/neutral_sample.wav',   'Standard baseline speech with stable pitch and moderate dynamics.', 0.00,  0.00),
    ('calm',      'Calm',      '😌', '/static/samples/calm_sample.wav',      'Relaxed vocal inflection, low energy, and smooth harmonic flow.',    0.45, -0.40),
    ('happy',     'Happy',     '😄', '/static/samples/happy_sample.wav',     'Elevated dynamic pitch variation, high spectral brightness.',        0.85,  0.65),
    ('sad',       'Sad',       '😢', '/static/samples/sad_sample.wav',       'Subdued loudness, downward pitch trend, lower spectral energy.',    -0.70, -0.60),
    ('angry',     'Angry',     '😡', '/static/samples/angry_sample.wav',     'Sharp attack, high vocal intensity, high RMS energy and harmonics.', -0.65,  0.90),
    ('fearful',   'Fearful',   '😨', '/static/samples/fearful_sample.wav',   'High fundamental pitch jitter, wide dynamic range, and vocal tension.', -0.75, 0.70),
    ('disgust',   'Disgust',   '🤢', '/static/samples/disgust_sample.wav',   'Guttural vocal quality, lower pitch with distinctive spectral tilt.', -0.80, -0.10),
    ('surprised', 'Surprised', '😲', '/static/samples/surprised_sample.wav', 'Sudden upward pitch leap and fast vocal onset.',                     0.35,  0.80)
ON CONFLICT (id) DO NOTHING;
