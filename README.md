# LLM Evaluation Hub

A universal LLM evaluation platform that combines RAGAS and LangSmith to provide comprehensive evaluation capabilities for RAG systems, agents, and general LLM applications.

## Features

- **Automated Testset Generation**: Generate synthetic QA pairs from documents using knowledge graphs
- **Multi-dimensional Evaluation**: Support for accuracy, factual correctness, precision, recall, F1, and more
- **LLM Judge System**: Use other LLMs to evaluate responses with custom criteria
- **Agent Evaluation**: Specialized metrics for agent systems including average turn and success rate
- **RAGAS Integration**: Leverage RAGAS metrics for RAG system evaluation
- **LangSmith Integration**: Use LangSmith for evaluation tracking and monitoring
- **RESTful API**: Complete API for programmatic access
- **Docker Support**: Easy deployment with Docker Compose

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key
- Google Gemini API key
- LangSmith API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-eval-hub
   ```

2. **Set up environment**
   ```bash
   make setup
   cp env.example .env
   ```

3. **Configure environment variables**
   Edit `.env` file with your API keys:
   ```bash
   LANGSMITH_API_KEY=lsv2_pt_your_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   SECRET_KEY=your-secret-key-here
   ```

4. **Run with Docker Compose**
   ```bash
   make up
   ```

5. **Access the application**
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

### Local Development

1. **Install dependencies**
   ```bash
   make install
   ```

2. **Run the application**
   ```bash
   make run
   ```

## Web Interface

The platform includes a modern web interface accessible at http://localhost:8000 with the following features:

### Dashboard
- System status monitoring
- Recent activity tracking
- Quick access to all features

### Scenario to Documents
- Convert scenario prompts to structured documents
- Specify output folder and number of documents
- Generate comprehensive documentation from descriptions

### RAG Testset Generation
- Load documents from folder using LangChain loaders
- Support for multiple file formats (PDF, TXT, DOCX, etc.)
- Configurable chunking with size and overlap settings
- Generate diverse question types (single-hop, multi-hop)

### Agent Testset Generation
- Create agent scenarios with goals and success criteria
- Export to Excel format with multiple sheets
- Include evaluation metrics and difficulty levels

### Evaluation
- Multi-testset evaluation support
- Multiple judge model selection (OpenAI, Gemini, Ollama, Hugging Face)
- Comprehensive metrics selection
- Real-time progress tracking

## API Usage

### Generate Documents from Scenario

```bash
# Convert scenario to documents
curl -X POST "http://localhost:8000/api/v1/testset/scenario-to-docs" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A student wants to learn about machine learning algorithms",
    "output_folder": "./data/generated_docs",
    "num_docs": 5
  }'
```

### Generate RAG Testset from Folder

```bash
# Generate RAG testset from documents folder
curl -X POST "http://localhost:8000/api/v1/testset/generate/rag/from-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "./data/documents",
    "testset_size": 10,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "single_hop_ratio": 0.5,
    "multi_hop_abstract_ratio": 0.25,
    "multi_hop_specific_ratio": 0.25
  }'
```

### Generate Agent Testset with Excel Output

```bash
# Generate agent testset with Excel output
curl -X POST "http://localhost:8000/api/v1/testset/generate/agent/excel" \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
        "name": "Customer Support",
        "description": "Help customer with product inquiry",
        "goal": "Resolve customer question",
        "expected_outcome": "Customer satisfied",
        "difficulty": "medium",
        "tools": ["search", "calculator"]
      }
    ],
    "output_path": "./results/agent_testsets"
  }'
```

### Evaluate System

```bash
# Evaluate RAG system
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "testset_id": "testset_20241201_120000",
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "metrics": ["accuracy", "factual_correctness", "precision", "recall", "f1"],
    "system_type": "rag"
  }'
```

### LLM Judge Evaluation

```bash
# Evaluate with OpenAI judge
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
    "judge_model_type": "openai",
    "judge_model": "gpt-4-turbo-preview",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'

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
    "judge_model": "gemini-pro",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'

# Evaluate with Ollama judge
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
    "judge_model_type": "ollama",
    "judge_model": "llama2",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'
```

## Multi-Model Judge Support

The platform supports multiple LLM models as judges for evaluation:

### Supported Judge Models

1. **OpenAI Models**
   - `gpt-4-turbo-preview`
   - `gpt-4`
   - `gpt-3.5-turbo`

2. **Google Gemini Models**
   - `gemini-pro`
   - `gemini-pro-vision`

3. **Ollama Models** (Local)
   - `llama2`
   - `codellama`
   - `mistral`
   - `neural-chat`

4. **Hugging Face Models**
   - `microsoft/DialoGPT-medium`
   - `facebook/blenderbot-400M-distill`

### Judge Model Selection

You can specify the judge model in your evaluation requests:

```json
{
  "judge_model_type": "gemini",
  "judge_model": "gemini-pro",
  "evaluation_criteria": ["accuracy", "factual_correctness"]
}
```

## Available Metrics

### RAG Metrics
- `accuracy`: Overall accuracy compared to ground truth
- `factual_correctness`: Factual accuracy of responses
- `precision`: Precision of information
- `recall`: Recall of relevant information
- `f1`: Harmonic mean of precision and recall
- `response_relevancy`: Relevance of responses to questions
- `faithfulness`: Faithfulness to provided context
- `context_precision`: Precision of retrieved context
- `context_recall`: Recall of relevant context
- `answer_relevancy`: Relevance of answers
- `answer_correctness`: Correctness of answers
- `answer_similarity`: Similarity to ground truth
- `semantic_similarity`: Semantic similarity
- `bleu_score`: BLEU score for text similarity
- `rouge_score`: ROUGE score for text similarity
- `exact_match`: Exact match with ground truth

### Agent Metrics
- `average_turn`: Average number of turns per conversation
- `success_rate`: Success rate of agent tasks
- `tool_call_accuracy`: Accuracy of tool calls
- `agent_goal_accuracy`: Accuracy in achieving goals
- `topic_adherence`: Adherence to conversation topic

## Project Structure

```
llm-eval-hub/
├── ai/                     # AI evaluation modules
│   ├── core/              # Core evaluation logic
│   │   ├── evaluator.py   # Main evaluator using RAGAS
│   │   └── llm_judge.py   # LLM judge system
│   └── testset/           # Testset generation
│       └── generator.py   # Testset generator service
├── api/                   # FastAPI application
│   ├── routes/           # API routes
│   │   ├── evaluation.py # Evaluation endpoints
│   │   ├── testset.py    # Testset endpoints
│   │   └── health.py     # Health check endpoints
│   ├── middleware.py     # Custom middleware
│   └── main.py          # FastAPI app
├── configs/             # Configuration
│   └── settings.py     # Application settings
├── data/               # Data storage
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
├── docs/              # Documentation
├── tutorials/         # Tutorial examples
├── examples/          # Usage examples
├── outputs/          # Inference results
├── artifacts/        # Model artifacts
├── results/          # Evaluation results
├── tests/            # Test files
├── docker-compose.yml # Docker configuration
├── Dockerfile        # Docker image
├── requirements.txt  # Python dependencies
├── pyproject.toml   # Project configuration
└── Makefile         # Build commands
```

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

### Building Docker Image

```bash
make build
```

## Configuration

The application can be configured through environment variables. See `env.example` for all available options.

### Key Configuration Options

- `LANGSMITH_API_KEY`: LangSmith API key for evaluation tracking
- `OPENAI_API_KEY`: OpenAI API key for LLM judge
- `GEMINI_API_KEY`: Google Gemini API key for judge evaluation
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `HUGGINGFACE_API_KEY`: Hugging Face API key for judge evaluation
- `SECRET_KEY`: Secret key for JWT token generation and session management
- `DATABASE_URL`: Database connection string
- `DEFAULT_EVALUATION_TIMEOUT`: Timeout for evaluations (seconds)
- `MAX_CONCURRENT_EVALUATIONS`: Maximum concurrent evaluations

## Monitoring

The application includes comprehensive monitoring:

- **Health Checks**: `/health`, `/health/detailed`, `/health/ready`, `/health/live`
- **Metrics**: Prometheus-compatible metrics endpoint
- **Structured Logging**: JSON-formatted logs with request tracing
- **Error Tracking**: Comprehensive error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review examples in `/examples` and `/tutorials`