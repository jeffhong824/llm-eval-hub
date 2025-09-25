#!/usr/bin/env python3
"""
Local development startup script for LLM Evaluation Hub.

This script starts the application locally without Docker.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import langchain
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        env_example = Path("env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your API keys")
        else:
            print("❌ env.example file not found")
            return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY", "LANGSMITH_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    
    print("✅ Environment variables configured")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/uploads",
        "data/raw", 
        "data/processed",
        "results",
        "artifacts",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def start_application():
    """Start the FastAPI application."""
    print("🚀 Starting LLM Evaluation Hub...")
    print("=" * 50)
    
    try:
        # Start the application
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function."""
    print("LLM Evaluation Hub - Local Development Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Setup environment
    if not setup_environment():
        return
    
    # Create directories
    create_directories()
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()



