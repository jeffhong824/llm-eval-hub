"""
Evaluation API Routes

Handles evaluation of RAG and Agent systems using external LLM APIs.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from ai.evaluation.rag_evaluator import RAGEvaluator, RAGTestCase
from ai.evaluation.agent_evaluator import AgentEvaluator, AgentTestCase
from ai.core.multi_judge import JudgeModelType
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()


class APIConfig(BaseModel):
    """Configuration for external LLM API."""
    url: str = Field(description="API endpoint URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation APIs")


class RAGEvaluationRequest(BaseModel):
    """Request model for RAG evaluation."""
    testset_path: str = Field(description="Path to RAG testset Excel file")
    api_config: APIConfig = Field(description="External LLM API configuration")
    judge_model_type: str = Field(description="Judge model type (openai, gemini, ollama, huggingface)")
    judge_model: str = Field(description="Judge model name")
    output_path: str = Field(description="Path to save evaluation results")
    selected_metrics: List[str] = Field(default_factory=list, description="Selected metrics to evaluate")


class AgentEvaluationRequest(BaseModel):
    """Request model for Agent evaluation."""
    testset_path: str = Field(description="Path to Agent testset Excel file")
    api_config: APIConfig = Field(description="External LLM API configuration")
    judge_model_type: str = Field(description="Judge model type (openai, gemini, ollama, huggingface)")
    judge_model: str = Field(description="Judge model name")
    output_path: str = Field(description="Path to save evaluation results")
    max_turns: int = Field(default=30, ge=1, le=100, description="Maximum conversation turns")
    selected_metrics: List[str] = Field(default_factory=list, description="Selected metrics to evaluate")


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    success: bool
    message: str
    output_path: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/rag", response_model=EvaluationResponse)
async def evaluate_rag_system(request: RAGEvaluationRequest):
    """Evaluate RAG system using external LLM API."""
    try:
        logger.info(f"Starting RAG evaluation with testset: {request.testset_path}")
        
        # Validate testset file exists
        testset_path = Path(request.testset_path)
        if not testset_path.exists():
            raise HTTPException(status_code=404, detail=f"Testset file not found: {request.testset_path}")
        
        # Validate judge model type
        try:
            judge_model_type = JudgeModelType(request.judge_model_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid judge model type: {request.judge_model_type}")
        
        # Create evaluator
        evaluator = RAGEvaluator(judge_model_type, request.judge_model)
        
        # Load testset
        test_cases = evaluator.load_testset(request.testset_path)
        if not test_cases:
            raise HTTPException(status_code=400, detail="No test cases found in testset")
        
        # Convert API config to dict
        api_config = request.api_config.dict()
        
        # Run evaluation
        results = await evaluator.evaluate_batch(test_cases, api_config)
        
        # Save results
        save_result = evaluator.save_results(results, request.output_path)
        
        if save_result['success']:
            return EvaluationResponse(
                success=True,
                message=f"RAG evaluation completed successfully! Evaluated {len(results)} test cases.",
                output_path=save_result['output_path'],
                summary=save_result['summary']
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to save results: {save_result['error']}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent", response_model=EvaluationResponse)
async def evaluate_agent_system(request: AgentEvaluationRequest):
    """Evaluate Agent system using external LLM API."""
    try:
        logger.info(f"Starting Agent evaluation with testset: {request.testset_path}")
        
        # Validate testset file exists
        testset_path = Path(request.testset_path)
        if not testset_path.exists():
            raise HTTPException(status_code=404, detail=f"Testset file not found: {request.testset_path}")
        
        # Validate judge model type
        try:
            judge_model_type = JudgeModelType(request.judge_model_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid judge model type: {request.judge_model_type}")
        
        # Create evaluator
        evaluator = AgentEvaluator(judge_model_type, request.judge_model)
        
        # Load testset
        test_cases = evaluator.load_testset(request.testset_path)
        if not test_cases:
            raise HTTPException(status_code=400, detail="No test cases found in testset")
        
        # Convert API config to dict
        api_config = request.api_config.dict()
        
        # Run evaluation
        results = await evaluator.evaluate_batch(test_cases, api_config, request.max_turns)
        
        # Save results
        save_result = evaluator.save_results(results, request.output_path)
        
        if save_result['success']:
            return EvaluationResponse(
                success=True,
                message=f"Agent evaluation completed successfully! Evaluated {len(results)} test cases.",
                output_path=save_result['output_path'],
                summary=save_result['summary']
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to save results: {save_result['error']}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Agent evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/available")
async def get_available_metrics():
    """Get list of available evaluation metrics."""
    rag_metrics = [
        "accuracy",
        "factual_correctness", 
        "precision",
        "recall",
        "f1",
        "response_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall"
    ]
    
    agent_metrics = [
        "task_completion",
        "conversation_quality",
        "user_satisfaction",
        "goal_achievement",
        "average_turn",
        "success_rate"
    ]
    
    return {
        "rag_metrics": rag_metrics,
        "agent_metrics": agent_metrics,
        "common_metrics": ["accuracy", "precision", "recall", "f1"]
    }


@router.get("/judge-models/available")
async def get_available_judge_models():
    """Get list of available judge models."""
    from ai.core.model_checker import ModelChecker
    
    model_checker = ModelChecker()
    available_models = model_checker.check_all_models()
    
    return {
        "openai": available_models.get("openai", []),
        "gemini": available_models.get("gemini", []),
        "ollama": available_models.get("ollama", []),
        "huggingface": available_models.get("huggingface", [])
    }


@router.post("/test-api-connection")
async def test_api_connection(api_config: APIConfig):
    """Test connection to external LLM API."""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test with a simple message
            test_data = {
                "message": "Hello, this is a test message.",
                "mode": "chat",
                "sessionId": api_config.session_id or "test"
            }
            
            async with session.post(
                api_config.url, 
                headers=api_config.headers, 
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "message": "API connection successful",
                        "response_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    }
                else:
                    return {
                        "success": False,
                        "message": f"API returned status {response.status}",
                        "error": await response.text()
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "message": "API connection failed",
            "error": str(e)
        }


@router.get("/status")
async def get_evaluation_status():
    """Get current evaluation system status."""
    return {
        "status": "ready",
        "supported_evaluation_types": ["rag", "agent"],
        "supported_judge_models": ["openai", "gemini", "ollama", "huggingface"],
        "max_concurrent_evaluations": 5,
        "default_max_turns": 30
    }