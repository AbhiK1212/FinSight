# Single stage build for production
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies and Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies globally (they'll be accessible to all users)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY env.example ./.env

# Create user and set up directories
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/data/models /app/logs && \
    chown -R app:app /app

# Switch to non-root user for security
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=5)" || exit 1

# Run the application with production settings
CMD ["uvicorn", "src.insight.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]
