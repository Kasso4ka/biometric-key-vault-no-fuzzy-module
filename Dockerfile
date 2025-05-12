FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY service-requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r service-requirements.txt

# Copy model files
COPY det.onnx .
COPY rec.onnx .
COPY binary_encoder_best.onnx .

# Copy application code
COPY utils.py .
COPY service.py .

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "10", "--timeout-keep-alive", "300", "--timeout", "600", "--h11-max-incomplete-event-size", "104857600"]