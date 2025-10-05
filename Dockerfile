# J.A.R.V.I.S. Dockerfile
# Multi-stage build for optimal containerization

# Stage 1: Builder stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    libasound2-dev \
    libjack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Create jarvis user for security
RUN useradd --create-home --shell /bin/bash jarvis

# Set working directory
WORKDIR /home/jarvis

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/jarvis/.local

# Copy application code
COPY . /home/jarvis/jarvis

# Set ownership
RUN chown -R jarvis:jarvis /home/jarvis

# Switch to jarvis user
USER jarvis

# Add local bin to PATH
ENV PATH="/home/jarvis/.local/bin:${PATH}"

# Set environment variables
ENV PYTHONPATH="/home/jarvis:${PYTHONPATH}"
ENV DISPLAY=:0
ENV QT_QPA_PLATFORM=offscreen

# Create necessary directories
RUN mkdir -p /home/jarvis/jarvis/data /home/jarvis/jarvis/logs /home/jarvis/jarvis/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/home/jarvis'); from core.jarvis import JARVIS; print('JARVIS health check passed')" || exit 1

# Default command
CMD ["python", "main.py", "--verbose"]

# Expose ports (if needed for web interface)
EXPOSE 8080

# Labels
LABEL maintainer="J.A.R.V.I.S. Team"
LABEL version="2.0"
LABEL description="Ultra-Advanced AI Personal Assistant"

# Volume mounts for persistent data
VOLUME ["/home/jarvis/jarvis/data", "/home/jarvis/jarvis/logs", "/home/jarvis/jarvis/config"]