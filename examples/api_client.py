"""
API client example for LLM Evaluation Hub.

This example demonstrates how to interact with the evaluation platform via API.
"""

import asyncio
import httpx
import json
from typing import Dict, List, Any


class LLMEvalClient:
    """Client for interacting with LLM Evaluation Hub API."""
    
    def __init__(self, base_url: str = "http://localhost:3010"):
        """Initialize the client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def generate_rag_testset(
        self,
        testset_size: int = 10,
        single_hop_ratio: float = 0.5,
        multi_hop_abstract_ratio: float = 0.25,
        multi_hop_specific_ratio: float = 0.25
    ) -> Dict[str, Any]:
        """Generate RAG testset."""
        payload = {
            "testset_size": testset_size,
            "single_hop_ratio": single_hop_ratio,
            "multi_hop_abstract_ratio": multi_hop_abstract_ratio,
            "multi_hop_specific_ratio": multi_hop_specific_ratio,
            "include_personas": True,
            "save_knowledge_graph": True
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/testset/generate/rag",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def generate_scenario_testset(
        self,
        scenarios: List[str],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate testset from scenarios."""
        payload = {
            "scenarios": scenarios,
            "config": config or {}
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/testset/generate/scenario",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def evaluate_system(
        self,
        testset_id: str,
        llm_endpoint: str,
        metrics: List[str],
        system_type: str = "rag",
        evaluation_name: str = None
    ) -> Dict[str, Any]:
        """Evaluate a system."""
        payload = {
            "testset_id": testset_id,
            "llm_endpoint": llm_endpoint,
            "metrics": metrics,
            "system_type": system_type,
            "evaluation_name": evaluation_name
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/evaluation/evaluate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def evaluate_with_judge(
        self,
        testset_data: List[Dict[str, Any]],
        llm_endpoint: str,
        evaluation_criteria: List[str] = None,
        judge_model: str = "gpt-4-turbo-preview"
    ) -> Dict[str, Any]:
        """Evaluate with LLM judge."""
        payload = {
            "testset_data": testset_data,
            "llm_endpoint": llm_endpoint,
            "evaluation_criteria": evaluation_criteria,
            "judge_model": judge_model,
            "temperature": 0.1
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/evaluation/judge",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_available_metrics(self) -> Dict[str, Any]:
        """Get available evaluation metrics."""
        response = await self.client.get(f"{self.base_url}/api/v1/evaluation/metrics/available")
        response.raise_for_status()
        return response.json()
    
    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation results."""
        response = await self.client.get(f"{self.base_url}/api/v1/evaluation/results/{evaluation_id}")
        response.raise_for_status()
        return response.json()
    
    async def list_evaluation_results(
        self,
        limit: int = 10,
        offset: int = 0,
        status: str = None
    ) -> Dict[str, Any]:
        """List evaluation results."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        response = await self.client.get(
            f"{self.base_url}/api/v1/evaluation/results",
            params=params
        )
        response.raise_for_status()
        return response.json()


async def example_rag_evaluation():
    """Example: Complete RAG evaluation workflow."""
    print("=== RAG Evaluation Example ===")
    
    client = LLMEvalClient()
    
    try:
        # Check API health
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Generate testset from scenarios
        scenarios = [
            "A student wants to learn about machine learning algorithms",
            "A professional needs to understand deep learning concepts",
            "A researcher is studying neural network architectures"
        ]
        
        print("Generating testset from scenarios...")
        testset_result = await client.generate_scenario_testset(scenarios)
        print(f"Generated testset: {testset_result['testset_id']}")
        
        # Evaluate system
        print("Evaluating system...")
        evaluation_result = await client.evaluate_system(
            testset_id=testset_result['testset_id'],
            llm_endpoint="https://api.openai.com/v1/chat/completions",
            metrics=["accuracy", "factual_correctness", "precision", "recall", "f1"],
            system_type="rag",
            evaluation_name="example_rag_evaluation"
        )
        
        print(f"Evaluation ID: {evaluation_result['evaluation_id']}")
        print(f"Status: {evaluation_result['status']}")
        print("Metrics:")
        for metric, score in evaluation_result['metrics'].items():
            print(f"  {metric}: {score:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_judge_evaluation():
    """Example: LLM judge evaluation."""
    print("\n=== LLM Judge Evaluation Example ===")
    
    client = LLMEvalClient()
    
    try:
        # Sample testset data
        testset_data = [
            {
                "question": "What is machine learning?",
                "ground_truth": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "context": "Machine learning is a subset of AI that focuses on algorithms...",
                "llm_response": "Machine learning is a method of data analysis that automates analytical model building."
            },
            {
                "question": "How does deep learning work?",
                "ground_truth": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "context": "Deep learning uses neural networks with multiple layers...",
                "llm_response": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers."
            }
        ]
        
        # Evaluate with judge
        print("Running LLM judge evaluation...")
        judge_result = await client.evaluate_with_judge(
            testset_data=testset_data,
            llm_endpoint="https://api.openai.com/v1/chat/completions",
            evaluation_criteria=["accuracy", "factual_correctness", "precision"],
            judge_model="gpt-4-turbo-preview"
        )
        
        print(f"Batch ID: {judge_result['batch_id']}")
        print(f"Status: {judge_result['status']}")
        print(f"Average Score: {judge_result['average_score']:.3f}")
        print(f"Successful Evaluations: {judge_result['successful_evaluations']}")
        print(f"Failed Evaluations: {judge_result['failed_evaluations']}")
        
        print("Score Distribution:")
        for range_name, count in judge_result['score_distribution'].items():
            print(f"  {range_name}: {count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_agent_evaluation():
    """Example: Agent evaluation."""
    print("\n=== Agent Evaluation Example ===")
    
    client = LLMEvalClient()
    
    try:
        # Generate agent testset
        agent_scenarios = [
            {
                "id": "scenario_1",
                "description": "Help user find information about machine learning",
                "type": "information_retrieval",
                "difficulty": "easy",
                "tools": ["search", "summarize"],
                "expected_outcome": "Provide accurate information about ML"
            },
            {
                "id": "scenario_2",
                "description": "Assist user in implementing a simple ML model",
                "type": "task_execution",
                "difficulty": "medium",
                "tools": ["code_generation", "explanation"],
                "expected_outcome": "Generate working code with explanations"
            }
        ]
        
        print("Generating agent testset...")
        testset_result = await client.generate_scenario_testset(
            scenarios=[s["description"] for s in agent_scenarios]
        )
        
        # Evaluate agent system
        print("Evaluating agent system...")
        evaluation_result = await client.evaluate_system(
            testset_id=testset_result['testset_id'],
            llm_endpoint="https://api.openai.com/v1/chat/completions",
            metrics=["accuracy", "factual_correctness", "average_turn", "success_rate"],
            system_type="agent",
            evaluation_name="example_agent_evaluation"
        )
        
        print(f"Agent Evaluation ID: {evaluation_result['evaluation_id']}")
        print(f"Status: {evaluation_result['status']}")
        print("Agent Metrics:")
        for metric, score in evaluation_result['metrics'].items():
            print(f"  {metric}: {score:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def example_metrics_exploration():
    """Example: Explore available metrics."""
    print("\n=== Metrics Exploration Example ===")
    
    client = LLMEvalClient()
    
    try:
        # Get available metrics
        metrics_info = await client.get_available_metrics()
        
        print("Available RAG Metrics:")
        for metric in metrics_info['rag_metrics']:
            print(f"  - {metric}")
        
        print("\nAvailable Agent Metrics:")
        for metric in metrics_info['agent_metrics']:
            print(f"  - {metric}")
        
        # List recent evaluations
        print("\nRecent Evaluations:")
        evaluations = await client.list_evaluation_results(limit=5)
        
        for evaluation in evaluations['results']:
            print(f"  - {evaluation.get('evaluation_id', 'N/A')}: {evaluation.get('status', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


async def main():
    """Run all examples."""
    print("LLM Evaluation Hub - API Client Examples")
    print("=" * 50)
    
    try:
        await example_rag_evaluation()
        await example_judge_evaluation()
        await example_agent_evaluation()
        await example_metrics_exploration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure the API server is running on http://localhost:3010")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())



