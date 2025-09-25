# Quick Start Guide

This guide will help you get the LLM Evaluation Hub running quickly.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional)
- OpenAI API key
- LangSmith API key

## Option 1: Docker Compose (Recommended)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd llm-eval-hub
cp env.example .env
```

### 2. Configure Environment

Edit `.env` file with your API keys:

```bash
LANGSMITH_API_KEY=your_langsmith_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your-secret-key-here
```

**Note about SECRET_KEY**: This is used for JWT token generation and session management. Generate a secure random string for production use.

### 3. Start Services

```bash
docker compose up -d
```

### 4. Access the Application

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Option 2: Local Development

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your API keys
```

### 3. Start the Application

```bash
python scripts/start.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Quick Test

### 1. Check Health

```bash
curl http://localhost:8000/health
```

### 2. Generate Testset

```bash
curl -X POST "http://localhost:8000/api/v1/testset/generate/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      "A student wants to learn about machine learning",
      "A professional needs to understand deep learning"
    ]
  }'
```

### 3. Evaluate System

```bash
# Evaluate with Gemini judge
curl -X POST "http://localhost:8000/api/v1/evaluation/judge" \
  -H "Content-Type: application/json" \
  -d '{
    "testset_data": [
      {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of AI...",
        "llm_response": "Machine learning is a method of data analysis..."
      }
    ],
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "judge_model_type": "gemini",
    "judge_model": "gemini-pro"
  }'
```

## Available Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /api/v1/testset/generate/rag` - Generate RAG testset
- `POST /api/v1/testset/generate/scenario` - Generate from scenarios
- `POST /api/v1/testset/generate/agent` - Generate agent testset
- `POST /api/v1/evaluation/evaluate` - Evaluate system
- `POST /api/v1/evaluation/judge` - Multi-model LLM judge evaluation
- `GET /api/v1/evaluation/judge/models` - Available judge models
- `GET /api/v1/evaluation/metrics/available` - Available metrics

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in .env file
   API_PORT=8001
   ```

2. **Missing API keys**
   ```bash
   # Make sure .env file has correct API keys
   LANGSMITH_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

3. **Docker issues**
   ```bash
   # Check if Docker is running
   docker --version
   docker compose --version
   ```

### Logs

View application logs:

```bash
# Docker Compose
docker compose logs -f

# Local development
# Logs are displayed in the terminal
```

## Next Steps

1. Explore the API documentation at http://localhost:8000/docs
2. Try the examples in the `examples/` directory
3. Run the tutorials in the `tutorials/` directory
4. Read the full documentation in `README.md`

## Support

If you encounter issues:

1. Check the logs for error messages
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Check the health endpoint for service status

For more help, create an issue in the repository.
