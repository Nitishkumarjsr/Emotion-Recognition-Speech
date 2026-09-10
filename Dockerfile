# ============================================================================
# VoxEmotion AI - Production Docker Container for Render / Cloud Platforms
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5002

WORKDIR /app

# Install system audio dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5002

# Start Gunicorn server
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120
