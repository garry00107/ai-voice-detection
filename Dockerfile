FROM python:3.11-slim

# Install FFmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY gradio_app.py ./
COPY .env ./

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Run Gradio app (better for demo) - can switch to FastAPI if needed
# For FastAPI: CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "gradio_app.py"]
