#!/bin/sh
# Docker healthcheck script for LLM Evaluation Hub
# Reads API_PORT from environment variable (defaults to 3010 if not set)

set -e

# Get port from environment variable, default to 3010
PORT=${API_PORT:-3010}

# Check health endpoint
curl -f "http://localhost:${PORT}/health" || exit 1

