#!/usr/bin/env python3
"""
Startup script for LLM Evaluation Hub.

This script provides a simple way to start the application with proper configuration.
"""

import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.settings import settings


def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        "LANGSMITH_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        settings.upload_dir,
        settings.results_dir,
        settings.artifacts_dir,
        "data/raw",
        "data/processed",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def main():
    """Main startup function."""
    print("🚀 Starting LLM Evaluation Hub")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Display configuration
    print(f"\n⚙️  Configuration:")
    print(f"   App Name: {settings.app_name}")
    print(f"   Version: {settings.app_version}")
    print(f"   Debug: {settings.debug}")
    print(f"   Host: {settings.api_host}")
    print(f"   Port: {settings.api_port}")
    print(f"   Log Level: {settings.log_level}")
    
    # Start the application
    print(f"\n🌐 Starting server...")
    print(f"   API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"   Health Check: http://{settings.api_host}:{settings.api_port}/health")
    print("\n" + "=" * 50)
    
    try:
        uvicorn.run(
            "api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()



