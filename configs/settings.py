"""Application configuration settings."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = Field(default="LLM Evaluation Hub", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=3010, env="API_PORT")
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    
    # LangSmith Configuration
    langsmith_api_key: str = Field(env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="llm-eval-hub", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", 
        env="LANGSMITH_ENDPOINT"
    )
    
    # LLM Judge Configuration
    # OpenAI Configuration
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Google Gemini Configuration
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    
    # Hugging Face Configuration
    huggingface_api_key: str = Field(default="", env="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(default="microsoft/DialoGPT-medium", env="HUGGINGFACE_MODEL")
    
    # Default Judge Model
    default_judge_model: str = Field(default="openai", env="DEFAULT_JUDGE_MODEL")
    
    # Database Configuration
    database_url: str = Field(env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Evaluation Configuration
    default_evaluation_timeout: int = Field(default=300, env="DEFAULT_EVALUATION_TIMEOUT")
    max_concurrent_evaluations: int = Field(default=10, env="MAX_CONCURRENT_EVALUATIONS")
    cache_evaluation_results: bool = Field(default=True, env="CACHE_EVALUATION_RESULTS")
    
    # Test Set Generation Configuration
    default_num_personas: int = Field(default=5, env="DEFAULT_NUM_PERSONAS")
    default_num_documents: int = Field(default=5, env="DEFAULT_NUM_DOCUMENTS")
    default_chunk_size: int = Field(default=5000, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=200, env="DEFAULT_CHUNK_OVERLAP")
    default_qa_per_chunk: int = Field(default=3, env="DEFAULT_QA_PER_CHUNK")
    default_tasks_per_chunk: int = Field(default=3, env="DEFAULT_TASKS_PER_CHUNK")
    default_max_turns: int = Field(default=30, env="DEFAULT_MAX_TURNS")
    default_language: str = Field(default="繁體中文", env="DEFAULT_LANGUAGE")
    
    # File Storage
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    results_dir: str = Field(default="./results", env="RESULTS_DIR")
    artifacts_dir: str = Field(default="./artifacts", env="ARTIFACTS_DIR")
    outputs_dir: str = Field(default="./outputs", env="OUTPUTS_DIR")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Available Evaluation Metrics
    available_metrics: List[str] = Field(
        default=[
            "accuracy",
            "factual_correctness", 
            "precision",
            "recall",
            "f1",
            "response_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
            "answer_correctness",
            "answer_similarity",
            "semantic_similarity",
            "bleu_score",
            "rouge_score",
            "exact_match"
        ]
    )
    
    # Agent-specific metrics
    agent_metrics: List[str] = Field(
        default=[
            "average_turn",
            "success_rate",
            "tool_call_accuracy",
            "agent_goal_accuracy",
            "topic_adherence"
        ]
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
