"""
Basic usage tutorial for LLM Evaluation Hub.

This tutorial demonstrates the basic functionality of the evaluation platform.
"""

import asyncio
import pandas as pd
from langchain.schema import Document

# Import the evaluation modules
from ai.core.evaluator import LLMEvaluator
from ai.core.llm_judge import LLMJudge, JudgeRequest, EvaluationCriteria
from ai.testset.generator import TestsetGeneratorService, TestsetConfig


async def basic_testset_generation():
    """Demonstrate basic testset generation."""
    print("=== Basic Testset Generation ===")
    
    # Initialize testset generator
    generator = TestsetGeneratorService("your-openai-api-key")
    
    # Sample documents
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"source": "ml_intro.txt", "topic": "machine_learning"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"source": "dl_intro.txt", "topic": "deep_learning"}
        )
    ]
    
    # Configure testset generation
    config = TestsetConfig(
        testset_size=5,
        single_hop_ratio=0.6,
        multi_hop_abstract_ratio=0.2,
        multi_hop_specific_ratio=0.2,
        include_personas=True,
        save_knowledge_graph=True
    )
    
    # Generate testset
    result = await generator.generate_rag_testset(documents, config)
    
    print(f"Generated testset ID: {result.testset_id}")
    print(f"Testset size: {result.testset_size}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Status: {result.status}")
    
    if result.status == "success":
        print("\nSample testset data:")
        print(result.testset_data.head())
    
    return result


async def basic_evaluation():
    """Demonstrate basic evaluation using RAGAS."""
    print("\n=== Basic Evaluation ===")
    
    # Initialize evaluator
    evaluator = LLMEvaluator()
    
    # Sample testset data
    testset_data = pd.DataFrame({
        "question": ["What is machine learning?", "How does deep learning work?"],
        "context": [
            "Machine learning is a subset of AI that focuses on algorithms...",
            "Deep learning uses neural networks with multiple layers..."
        ],
        "answer": [
            "Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers to model complex patterns."
        ],
        "ground_truth": [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
        ]
    })
    
    # Define metrics to evaluate
    metrics = ["accuracy", "factual_correctness", "precision", "recall", "f1"]
    
    # Evaluate system
    result = await evaluator.evaluate_rag_system(
        testset_data=testset_data,
        llm_endpoint="https://api.openai.com/v1/chat/completions",
        metrics=metrics,
        evaluation_name="basic_evaluation"
    )
    
    print(f"Evaluation ID: {result.evaluation_id}")
    print(f"Status: {result.status}")
    print(f"Evaluation time: {result.evaluation_time:.2f}s")
    
    if result.status == "success":
        print("\nEvaluation metrics:")
        for metric, score in result.metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    return result


async def basic_llm_judge():
    """Demonstrate LLM judge evaluation."""
    print("\n=== LLM Judge Evaluation ===")
    
    # Initialize LLM judge
    judge = LLMJudge("your-openai-api-key")
    
    # Get default evaluation criteria
    default_criteria = judge.get_default_criteria()
    
    # Create judge request
    request = JudgeRequest(
        question="What is machine learning?",
        ground_truth="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        context="Machine learning is a subset of AI that focuses on algorithms...",
        llm_response="Machine learning is a method of data analysis that automates analytical model building.",
        evaluation_criteria=list(default_criteria.values()),
        judge_model="gpt-4-turbo-preview",
        temperature=0.1
    )
    
    # Evaluate single response
    result = await judge.evaluate_single_response(request)
    
    print(f"Overall score: {result.overall_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nCriteria scores:")
    for criteria, score in result.criteria_scores.items():
        print(f"  {criteria}: {score:.3f}")
    
    return result


async def scenario_based_generation():
    """Demonstrate scenario-based testset generation."""
    print("\n=== Scenario-based Generation ===")
    
    # Initialize testset generator
    generator = TestsetGeneratorService("your-openai-api-key")
    
    # User-provided scenarios
    scenarios = [
        "A student wants to understand the basics of machine learning",
        "A professional needs to implement a recommendation system",
        "A researcher is studying neural network architectures"
    ]
    
    # Generate testset from scenarios
    result = await generator.generate_from_user_scenarios(scenarios)
    
    print(f"Generated testset ID: {result.testset_id}")
    print(f"Testset size: {result.testset_size}")
    print(f"Status: {result.status}")
    
    if result.status == "success":
        print("\nGenerated testset data:")
        print(result.testset_data.head())
    
    return result


async def agent_evaluation():
    """Demonstrate agent evaluation."""
    print("\n=== Agent Evaluation ===")
    
    # Initialize evaluator
    evaluator = LLMEvaluator()
    
    # Sample agent scenarios
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
    
    # Convert to testset data
    testset_data = pd.DataFrame([
        {
            "question": "Help me understand machine learning",
            "context": "User is a beginner",
            "expected_answer": "Machine learning is a subset of AI...",
            "scenario_type": "information_retrieval",
            "difficulty": "easy",
            "tools_required": ["search", "summarize"],
            "expected_turns": 3
        },
        {
            "question": "Show me how to implement a simple classifier",
            "context": "User has basic programming knowledge",
            "expected_answer": "Here's a simple example using scikit-learn...",
            "scenario_type": "task_execution", 
            "difficulty": "medium",
            "tools_required": ["code_generation", "explanation"],
            "expected_turns": 5
        }
    ])
    
    # Evaluate agent system
    result = await evaluator.evaluate_agent_system(
        testset_data=testset_data,
        agent_endpoint="https://api.openai.com/v1/chat/completions",
        metrics=["accuracy", "factual_correctness", "average_turn", "success_rate"],
        evaluation_name="agent_evaluation"
    )
    
    print(f"Agent evaluation ID: {result.evaluation_id}")
    print(f"Status: {result.status}")
    
    if result.status == "success":
        print("\nAgent evaluation metrics:")
        for metric, score in result.metrics.items():
            print(f"  {metric}: {score:.3f}")
    
    return result


async def main():
    """Run all tutorials."""
    print("LLM Evaluation Hub - Basic Usage Tutorial")
    print("=" * 50)
    
    try:
        # Run tutorials
        await basic_testset_generation()
        await basic_evaluation()
        await basic_llm_judge()
        await scenario_based_generation()
        await agent_evaluation()
        
        print("\n" + "=" * 50)
        print("All tutorials completed successfully!")
        
    except Exception as e:
        print(f"\nError running tutorials: {str(e)}")
        print("Make sure to set your OpenAI API key in the code.")


if __name__ == "__main__":
    # Run the tutorial
    asyncio.run(main())



