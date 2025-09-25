"""
Multi-model judge example for LLM Evaluation Hub.

This example demonstrates how to use different judge models (OpenAI, Gemini, Ollama, Hugging Face).
"""

import asyncio
import httpx
import json
from typing import Dict, List, Any


class MultiJudgeClient:
    """Client for multi-model judge evaluation."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
    
    async def get_available_judge_models(self) -> Dict[str, Any]:
        """Get available judge models."""
        response = await self.client.get(f"{self.base_url}/api/v1/evaluation/judge/models")
        response.raise_for_status()
        return response.json()
    
    async def evaluate_with_judge(
        self,
        testset_data: List[Dict[str, Any]],
        judge_model_type: str = "openai",
        judge_model: str = None,
        evaluation_criteria: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate with specified judge model."""
        payload = {
            "testset_data": testset_data,
            "llm_endpoint": "https://api.openai.com/v1/chat/completions",
            "evaluation_criteria": evaluation_criteria,
            "judge_model_type": judge_model_type,
            "judge_model": judge_model,
            "temperature": 0.1
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/evaluation/judge",
            json=payload
        )
        response.raise_for_status()
        return response.json()


async def example_openai_judge():
    """Example: Using OpenAI as judge."""
    print("=== OpenAI Judge Example ===")
    
    client = MultiJudgeClient()
    
    try:
        # Sample testset data
        testset_data = [
            {
                "question": "What is machine learning?",
                "ground_truth": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "context": "Machine learning is a subset of AI that focuses on algorithms...",
                "llm_response": "Machine learning is a method of data analysis that automates analytical model building."
            }
        ]
        
        # Evaluate with OpenAI judge
        print("Running OpenAI judge evaluation...")
        result = await client.evaluate_with_judge(
            testset_data=testset_data,
            judge_model_type="openai",
            judge_model="gpt-4-turbo-preview",
            evaluation_criteria=["accuracy", "factual_correctness"]
        )
        
        print(f"Judge Model Used: OpenAI GPT-4")
        print(f"Average Score: {result['average_score']:.3f}")
        print(f"Successful Evaluations: {result['successful_evaluations']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_gemini_judge():
    """Example: Using Gemini as judge."""
    print("\n=== Gemini Judge Example ===")
    
    client = MultiJudgeClient()
    
    try:
        # Sample testset data
        testset_data = [
            {
                "question": "What is deep learning?",
                "ground_truth": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "context": "Deep learning uses neural networks with multiple layers...",
                "llm_response": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers."
            }
        ]
        
        # Evaluate with Gemini judge
        print("Running Gemini judge evaluation...")
        result = await client.evaluate_with_judge(
            testset_data=testset_data,
            judge_model_type="gemini",
            judge_model="gemini-pro",
            evaluation_criteria=["accuracy", "precision", "recall"]
        )
        
        print(f"Judge Model Used: Gemini Pro")
        print(f"Average Score: {result['average_score']:.3f}")
        print(f"Successful Evaluations: {result['successful_evaluations']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_ollama_judge():
    """Example: Using Ollama as judge."""
    print("\n=== Ollama Judge Example ===")
    
    client = MultiJudgeClient()
    
    try:
        # Sample testset data
        testset_data = [
            {
                "question": "How does a neural network work?",
                "ground_truth": "A neural network processes information through interconnected nodes (neurons) organized in layers.",
                "context": "Neural networks are computing systems inspired by biological neural networks...",
                "llm_response": "Neural networks work by processing data through layers of interconnected nodes that can learn patterns."
            }
        ]
        
        # Evaluate with Ollama judge
        print("Running Ollama judge evaluation...")
        result = await client.evaluate_with_judge(
            testset_data=testset_data,
            judge_model_type="ollama",
            judge_model="llama2",
            evaluation_criteria=["accuracy", "factual_correctness"]
        )
        
        print(f"Judge Model Used: Ollama Llama2")
        print(f"Average Score: {result['average_score']:.3f}")
        print(f"Successful Evaluations: {result['successful_evaluations']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_huggingface_judge():
    """Example: Using Hugging Face as judge."""
    print("\n=== Hugging Face Judge Example ===")
    
    client = MultiJudgeClient()
    
    try:
        # Sample testset data
        testset_data = [
            {
                "question": "What is natural language processing?",
                "ground_truth": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
                "context": "NLP is a branch of artificial intelligence that helps computers understand human language...",
                "llm_response": "NLP is a technology that enables computers to understand and process human language."
            }
        ]
        
        # Evaluate with Hugging Face judge
        print("Running Hugging Face judge evaluation...")
        result = await client.evaluate_with_judge(
            testset_data=testset_data,
            judge_model_type="huggingface",
            judge_model="microsoft/DialoGPT-medium",
            evaluation_criteria=["accuracy", "factual_correctness"]
        )
        
        print(f"Judge Model Used: Hugging Face DialoGPT")
        print(f"Average Score: {result['average_score']:.3f}")
        print(f"Successful Evaluations: {result['successful_evaluations']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_compare_judges():
    """Example: Compare different judge models."""
    print("\n=== Judge Comparison Example ===")
    
    client = MultiJudgeClient()
    
    try:
        # Get available models
        models_info = await client.get_available_judge_models()
        print("Available Judge Models:")
        for model_type, models in models_info['available_models'].items():
            print(f"  {model_type}: {models}")
        
        # Sample testset data
        testset_data = [
            {
                "question": "What is artificial intelligence?",
                "ground_truth": "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
                "context": "AI is a broad field of computer science...",
                "llm_response": "AI is the development of computer systems that can perform tasks typically requiring human intelligence."
            }
        ]
        
        # Compare different judges
        judges_to_compare = [
            ("openai", "gpt-4-turbo-preview"),
            ("gemini", "gemini-pro"),
            ("ollama", "llama2")
        ]
        
        results = {}
        
        for judge_type, model in judges_to_compare:
            try:
                print(f"\nTesting {judge_type}:{model}...")
                result = await client.evaluate_with_judge(
                    testset_data=testset_data,
                    judge_model_type=judge_type,
                    judge_model=model,
                    evaluation_criteria=["accuracy", "factual_correctness"]
                )
                
                results[f"{judge_type}:{model}"] = {
                    "average_score": result['average_score'],
                    "successful_evaluations": result['successful_evaluations'],
                    "evaluation_time": result['evaluation_time']
                }
                
                print(f"  Average Score: {result['average_score']:.3f}")
                print(f"  Evaluation Time: {result['evaluation_time']:.2f}s")
                
            except Exception as e:
                print(f"  Error with {judge_type}:{model}: {str(e)}")
                results[f"{judge_type}:{model}"] = {"error": str(e)}
        
        # Summary
        print("\n=== Comparison Summary ===")
        for judge, result in results.items():
            if "error" in result:
                print(f"{judge}: Error - {result['error']}")
            else:
                print(f"{judge}: Score {result['average_score']:.3f}, Time {result['evaluation_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def main():
    """Run all examples."""
    print("LLM Evaluation Hub - Multi-Model Judge Examples")
    print("=" * 60)
    
    try:
        await example_openai_judge()
        await example_gemini_judge()
        await example_ollama_judge()
        await example_huggingface_judge()
        await example_compare_judges()
        
        print("\n" + "=" * 60)
        print("All multi-model judge examples completed!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure the API server is running and all judge models are configured.")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())



