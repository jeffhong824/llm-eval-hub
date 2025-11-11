FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy entrypoint and healthcheck scripts
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY docker-healthcheck.sh /app/docker-healthcheck.sh

# Create necessary directories
RUN mkdir -p data/uploads results artifacts && \
    chmod +x /app/docker-entrypoint.sh /app/docker-healthcheck.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports (default port, actual port is controlled by API_PORT env var in docker-compose)
# Note: EXPOSE is mostly documentation - actual port mapping is done in docker-compose.yml
EXPOSE 3010 9090

# Health check - use the healthcheck script which reads API_PORT from env at runtime
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD /app/docker-healthcheck.sh

# Run the application using the entrypoint script
CMD ["/app/docker-entrypoint.sh"]



