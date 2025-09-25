"""
Example usage of the Evaluation system.

This example demonstrates how to use the RAG and Agent evaluation systems
with external LLM APIs.
"""

import asyncio
import json
from pathlib import Path

# Example API configurations
ANYTHINGLLM_CONFIG = {
    "url": "http://36.231.184.28:3001/api/v1/workspace/my-workspace/chat",
    "headers": {
        "Authorization": "Bearer 69GJRE6-1JT4ZTD-HC0VT40-A29C9VB",
        "Content-Type": "application/json"
    },
    "session_id": "1"
}

CUSTOM_CONVERSATION_CONFIG = {
    "url": "http://localhost:8000/conversations",
    "headers": {
        "Content-Type": "application/json"
    }
}

async def test_rag_evaluation():
    """Test RAG evaluation with external API."""
    print("Testing RAG Evaluation...")
    
    # Example RAG evaluation request
    rag_request = {
        "testset_path": "./outputs/rag_testset.xlsx",
        "api_config": ANYTHINGLLM_CONFIG,
        "judge_model_type": "openai",
        "judge_model": "gpt-3.5-turbo",
        "output_path": "./outputs/rag_evaluation_results.xlsx",
        "selected_metrics": [
            "accuracy",
            "factual_correctness",
            "precision",
            "recall",
            "f1"
        ]
    }
    
    print("RAG Evaluation Request:")
    print(json.dumps(rag_request, indent=2))
    
    # In a real scenario, you would make an HTTP request to the API
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(
    #         "http://localhost:8000/api/v1/evaluation/rag",
    #         json=rag_request
    #     ) as response:
    #         result = await response.json()
    #         print("RAG Evaluation Result:", result)

async def test_agent_evaluation():
    """Test Agent evaluation with external API."""
    print("\nTesting Agent Evaluation...")
    
    # Example Agent evaluation request
    agent_request = {
        "testset_path": "./outputs/agent_testset.xlsx",
        "api_config": CUSTOM_CONVERSATION_CONFIG,
        "judge_model_type": "gemini",
        "judge_model": "gemini-1.5-pro",
        "output_path": "./outputs/agent_evaluation_results.xlsx",
        "max_turns": 30,
        "selected_metrics": [
            "task_completion",
            "conversation_quality",
            "user_satisfaction",
            "goal_achievement",
            "success_rate"
        ]
    }
    
    print("Agent Evaluation Request:")
    print(json.dumps(agent_request, indent=2))
    
    # In a real scenario, you would make an HTTP request to the API
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(
    #         "http://localhost:8000/api/v1/evaluation/agent",
    #         json=agent_request
    #     ) as response:
    #         result = await response.json()
    #         print("Agent Evaluation Result:", result)

async def test_api_connection():
    """Test API connection."""
    print("\nTesting API Connection...")
    
    # Example API connection test
    connection_test = {
        "url": "http://36.231.184.28:3001/api/v1/workspace/my-workspace/chat",
        "headers": {
            "Authorization": "Bearer 69GJRE6-1JT4ZTD-HC0VT40-A29C9VB",
            "Content-Type": "application/json"
        }
    }
    
    print("API Connection Test Request:")
    print(json.dumps(connection_test, indent=2))
    
    # In a real scenario, you would make an HTTP request to the API
    # import aiohttp
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(
    #         "http://localhost:8000/api/v1/evaluation/test-api-connection",
    #         json=connection_test
    #     ) as response:
    #         result = await response.json()
    #         print("API Connection Test Result:", result)

def print_usage_examples():
    """Print usage examples for different API types."""
    print("\n" + "="*60)
    print("EVALUATION SYSTEM USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. AnythingLLM API Example:")
    print("""
    curl -X 'POST' \\
      'http://36.231.184.28:3001/api/v1/workspace/my-workspace/chat' \\
      -H 'accept: application/json' \\
      -H 'Authorization: Bearer 69GJRE6-1JT4ZTD-HC0VT40-A29C9VB' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "message": "What is AnythingLLM?",
        "mode": "chat",
        "sessionId": "1"
      }'
    """)
    
    print("\n2. Custom Conversation API Example:")
    print("""
    # Create conversation
    curl -X 'POST' \\
      'http://localhost:8000/conversations' \\
      -H 'Content-Type: application/json' \\
      -d '{"role": "buyer"}'
    
    # Send message
    curl -X 'POST' \\
      'http://localhost:8000/conversations/conv_123456/chat' \\
      -H 'Content-Type: application/json' \\
      -d '{"text": "我想要找信義區的三房兩廳，預算2500萬"}'
    """)
    
    print("\n3. RAG Evaluation API Call:")
    print("""
    curl -X 'POST' \\
      'http://localhost:8000/api/v1/evaluation/rag' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "testset_path": "./outputs/rag_testset.xlsx",
        "api_config": {
          "url": "http://36.231.184.28:3001/api/v1/workspace/my-workspace/chat",
          "headers": {
            "Authorization": "Bearer 69GJRE6-1JT4ZTD-HC0VT40-A29C9VB"
          }
        },
        "judge_model_type": "openai",
        "judge_model": "gpt-3.5-turbo",
        "output_path": "./outputs/rag_evaluation_results.xlsx"
      }'
    """)
    
    print("\n4. Agent Evaluation API Call:")
    print("""
    curl -X 'POST' \\
      'http://localhost:8000/api/v1/evaluation/agent' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "testset_path": "./outputs/agent_testset.xlsx",
        "api_config": {
          "url": "http://localhost:8000/conversations",
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "judge_model_type": "gemini",
        "judge_model": "gemini-1.5-pro",
        "output_path": "./outputs/agent_evaluation_results.xlsx",
        "max_turns": 30
      }'
    """)

async def main():
    """Main function to run all examples."""
    print("LLM Evaluation Hub - Evaluation Examples")
    print("="*50)
    
    await test_rag_evaluation()
    await test_agent_evaluation()
    await test_api_connection()
    
    print_usage_examples()
    
    print("\n" + "="*60)
    print("EVALUATION SYSTEM FEATURES")
    print("="*60)
    print("✅ RAG Evaluation: One-question-one-answer evaluation")
    print("✅ Agent Evaluation: Multi-turn conversation evaluation")
    print("✅ Multiple Judge Models: OpenAI, Gemini, Ollama, Hugging Face")
    print("✅ External API Support: Any LLM API with JSON responses")
    print("✅ Configurable Metrics: Select specific metrics to evaluate")
    print("✅ Excel Output: Detailed results saved to Excel files")
    print("✅ API Connection Testing: Test external APIs before evaluation")
    print("✅ Frontend Interface: Easy-to-use web interface")
    print("✅ Batch Processing: Evaluate multiple test cases efficiently")

if __name__ == "__main__":
    asyncio.run(main())


