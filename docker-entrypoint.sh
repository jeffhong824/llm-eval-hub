#!/bin/sh
# Docker entrypoint script for LLM Evaluation Hub
# Reads API_PORT from environment variable (defaults to 3010 if not set)

set -e

# Get port from environment variable, default to 3010
PORT=${API_PORT:-3010}

# Start uvicorn server
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"

