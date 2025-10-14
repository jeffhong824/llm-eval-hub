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


class CheckModelsRequest(BaseModel):
    """Request model for checking available models with custom API keys."""
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None


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
    """Get list of available judge models using .env API keys."""
    from ai.core.model_checker import ModelChecker
    
    model_checker = ModelChecker()
    available_models = await model_checker.check_all_models()
    
    return {
        "openai": available_models.get("openai", []),
        "gemini": available_models.get("gemini", []),
        "ollama": available_models.get("ollama", []),
        "huggingface": available_models.get("huggingface", [])
    }


@router.post("/judge-models/check-custom")
async def check_custom_judge_models(request: CheckModelsRequest):
    """Check available models with custom API keys or .env keys."""
    import httpx
    from configs.settings import settings
    
    result = {
        "openai": [],
        "gemini": [],
        "ollama": []
    }
    
    # Check OpenAI models (use custom key or .env key)
    openai_key = request.openai_api_key or settings.openai_api_key
    if openai_key:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    # Filter for GPT models suitable for evaluation (chat models only)
                    non_chat_keywords = ['davinci', 'curie', 'babbage', 'ada', 'instruct', 'embedding', 'whisper', 'dall-e', 'tts', 'transcribe']
                    gpt_models = [
                        model["id"] for model in data.get("data", [])
                        if "gpt" in model["id"].lower() and 
                        any(x in model["id"] for x in ["gpt-4", "gpt-3.5"]) and
                        not any(non_chat in model["id"].lower() for non_chat in non_chat_keywords)
                    ]
                    result["openai"] = sorted(gpt_models, reverse=True)[:15]  # Top 15 chat models
                else:
                    result["openai"] = [f"Error: {response.status_code}"]
        except Exception as e:
            result["openai"] = [f"Error: {str(e)}"]
    
    # Check Gemini models (use custom key or .env key)
    gemini_key = request.gemini_api_key or settings.gemini_api_key
    if gemini_key:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}",
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    gemini_models = [
                        model["name"].replace("models/", "") 
                        for model in data.get("models", [])
                        if "gemini" in model["name"].lower()
                    ]
                    result["gemini"] = gemini_models
                else:
                    result["gemini"] = [f"Error: {response.status_code}"]
        except Exception as e:
            result["gemini"] = [f"Error: {str(e)}"]
    
    # Check Ollama models (use custom URL or .env URL)
    ollama_url = request.ollama_base_url or settings.ollama_base_url
    if ollama_url:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ollama_url}/api/tags",
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    ollama_models = [model["name"] for model in data.get("models", [])]
                    result["ollama"] = ollama_models
                else:
                    result["ollama"] = [f"Error: {response.status_code}"]
        except Exception as e:
            result["ollama"] = [f"Error: {str(e)}"]
    
    return result


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


class WorkflowEvaluationRequest(BaseModel):
    """Request model for workflow evaluation."""
    testset_path: str
    target_api_url: str
    target_api_method: str = "POST"
    target_api_key_header: Optional[str] = None
    target_api_key_value: Optional[str] = None
    target_api_question_field: str = "question"
    target_api_response_field: str = "response"
    judge_model_provider: str = "openai"
    judge_model_name: str = "gpt-4-0613"
    # Dynamic API Keys for Judge LLM
    judge_openai_api_key: Optional[str] = None
    judge_gemini_api_key: Optional[str] = None
    judge_ollama_base_url: Optional[str] = None
    metrics: List[str] = Field(default_factory=lambda: ["answer_accuracy"])
    output_folder: str
    scenario_name: str


@router.post("/workflow/run-evaluation")
async def run_workflow_evaluation(request: WorkflowEvaluationRequest):
    """
    Run evaluation on a testset using user's API and RAGAS metrics.
    
    This endpoint:
    1. Loads the testset from Excel
    2. Calls the user's API for each question to get responses
    3. Uses RAGAS with GPT-4 to evaluate the responses
    4. Returns evaluation scores and saves results
    """
    try:
        import pandas as pd
        import requests
        import time
        from datetime import datetime
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from ragas import evaluate, aevaluate
        # Import RAGAS 0.3.6 metrics (all are classes that need instantiation)
        from ragas.metrics import (
            AnswerAccuracy,
            AnswerCorrectness,
            AnswerRelevancy,
            AnswerSimilarity,
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
            Faithfulness,
            FactualCorrectness,
            ResponseRelevancy
        )
        from datasets import Dataset
        from configs.settings import settings
        import os
        
        start_time = time.time()
        
        logger.info(f"Starting workflow evaluation")
        logger.info(f"Testset path received: {request.testset_path}")
        logger.info(f"Target API URL: {request.target_api_url}")
        logger.info(f"Selected metrics: {request.metrics}")
        
        # Step 1: Load testset
        if not os.path.exists(request.testset_path):
            logger.error(f"Testset file not found: {request.testset_path}")
            raise FileNotFoundError(f"測試集檔案不存在: {request.testset_path}")
        
        df = pd.read_excel(request.testset_path)
        logger.info(f"Loaded {len(df)} test cases from {request.testset_path}")
        
        # Step 2: Call user's API to get responses
        logger.info(f"Calling user API: {request.target_api_url}")
        
        responses = []
        for idx, row in df.iterrows():
            question = row.get('question', '')
            
            # Build API request
            headers = {'Content-Type': 'application/json'}
            if request.target_api_key_header and request.target_api_key_value:
                headers[request.target_api_key_header] = request.target_api_key_value
            
            payload = {request.target_api_question_field: question}
            
            # Add mode field for AnythingLLM compatibility
            if 'anythingllm' in request.target_api_url.lower() or 'workspace' in request.target_api_url.lower():
                payload['mode'] = 'chat'
            
            try:
                if request.target_api_method == "POST":
                    api_response = requests.post(
                        request.target_api_url,
                        json=payload,
                        headers=headers,
                        timeout=60  # Increase timeout for LLM responses
                    )
                else:
                    api_response = requests.get(
                        request.target_api_url,
                        params=payload,
                        headers=headers,
                        timeout=30
                    )
                
                if api_response.status_code == 200:
                    response_data = api_response.json()
                    
                    # Extract response based on field path
                    response_text = response_data
                    for field in request.target_api_response_field.split('.'):
                        if isinstance(response_text, dict):
                            response_text = response_text.get(field, '')
                        else:
                            break
                    
                    responses.append(str(response_text))
                    logger.info(f"Got response for question {idx+1}/{len(df)}")
                else:
                    logger.error(f"API call failed for question {idx}: {api_response.status_code}")
                    responses.append("")
                    
            except Exception as e:
                logger.error(f"Error calling API for question {idx}: {str(e)}")
                responses.append("")
        
        # Add responses to dataframe
        df['response'] = responses
        
        # Step 3: Validate and Initialize Judge LLM (use custom API keys if provided)
        logger.info(f"Initializing Judge LLM: {request.judge_model_provider}/{request.judge_model_name}")
        logger.info(f"Judge model details - Provider: {request.judge_model_provider}, Model: '{request.judge_model_name}'")
        
        if request.judge_model_provider == "openai":
            # Use custom API key if provided, otherwise use settings
            api_key = request.judge_openai_api_key or settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key is required. Please provide it in the form or set OPENAI_API_KEY in .env")
            
            # Validate model supports chat completions
            model_name = request.judge_model_name
            non_chat_models = ['text-davinci-003', 'text-davinci-002', 'davinci', 'curie', 'babbage', 'ada']
            if any(non_chat in model_name.lower() for non_chat in non_chat_models):
                raise ValueError(
                    f"模型 '{model_name}' 不支援聊天模式。\n\n"
                    f"請選擇以下支援聊天的模型：\n"
                    f"• gpt-4-turbo\n"
                    f"• gpt-4-0613\n"
                    f"• gpt-4\n"
                    f"• gpt-3.5-turbo\n"
                    f"• gpt-3.5-turbo-16k"
                )
            
            # 使用RAGAS推薦的新方式
            try:
                from ragas.llms.base import llm_factory
                evaluator_llm = llm_factory('openai', model=model_name, api_key=api_key)
                logger.info(f"Using RAGAS llm_factory for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to use llm_factory, falling back to LangchainLLMWrapper: {e}")
                # 回退到舊方式，但使用最小參數集
                chat_llm = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=api_key
                )
                evaluator_llm = LangchainLLMWrapper(chat_llm)
        elif request.judge_model_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Use custom API key if provided, otherwise use settings
            api_key = request.judge_gemini_api_key or settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key is required. Please provide it in the form or set GEMINI_API_KEY in .env")
            
            try:
                from ragas.llms.base import llm_factory
                evaluator_llm = llm_factory('google', model=request.judge_model_name, api_key=api_key)
                logger.info(f"Using RAGAS llm_factory for Gemini {request.judge_model_name}")
            except Exception as e:
                logger.warning(f"Failed to use llm_factory for Gemini, falling back to LangchainLLMWrapper: {e}")
                chat_llm = ChatGoogleGenerativeAI(
                    model=request.judge_model_name,
                    google_api_key=api_key,
                    temperature=0
                )
                evaluator_llm = LangchainLLMWrapper(chat_llm)
        elif request.judge_model_provider == "ollama":
            # Use custom base URL if provided, otherwise use settings
            base_url = request.judge_ollama_base_url or settings.ollama_base_url
            if not base_url:
                raise ValueError("Ollama base URL is required. Please provide it in the form or set OLLAMA_BASE_URL in .env")
            
            try:
                from ragas.llms.base import llm_factory
                evaluator_llm = llm_factory('ollama', model=request.judge_model_name, base_url=base_url)
                logger.info(f"Using RAGAS llm_factory for Ollama {request.judge_model_name}")
            except Exception as e:
                logger.warning(f"Failed to use llm_factory for Ollama, falling back to LangchainLLMWrapper: {e}")
                chat_llm = ChatOpenAI(
                    model_name=request.judge_model_name,
                    base_url=base_url,
                    api_key="ollama",
                    temperature=0
                )
                evaluator_llm = LangchainLLMWrapper(chat_llm)
        else:
            raise ValueError(f"Unsupported judge model provider: {request.judge_model_provider}")
        
        # Step 4: Prepare dataset for RAGAS
        evaluation_data = []
        for _, row in df.iterrows():
            data_point = {
                "question": row.get('question', ''),
                "answer": row.get('response', ''),  # The response from user's API
                "ground_truth": row.get('answer', row.get('reference', ''))  # Reference answer
            }
            
            # Add context if available
            if 'context' in row and pd.notna(row['context']):
                data_point['contexts'] = [str(row['context'])]
            
            evaluation_data.append(data_point)
        
        evaluation_dataset = Dataset.from_list(evaluation_data)
        
        # Step 5: Select metrics (instantiate metric classes)
        selected_metrics = []
        metric_map = {
            # Core metrics - 用於基本評估
            'answer_accuracy': AnswerAccuracy(),
            'answer_correctness': AnswerCorrectness(),
            'factual_correctness': FactualCorrectness(),
            'context_precision': ContextPrecision(),
            'context_recall': ContextRecall(),
            
            # Advanced metrics - 用於深度評估
            'faithfulness': Faithfulness(),
            'answer_relevancy': AnswerRelevancy(),
            'answer_similarity': AnswerSimilarity(),
            'context_relevance': ContextRelevance(),
            'response_relevancy': ResponseRelevancy()
        }
        
        for metric_name in request.metrics:
            if metric_name in metric_map:
                selected_metrics.append(metric_map[metric_name])
        
        if not selected_metrics:
            selected_metrics = [AnswerAccuracy()]  # Default metric
        
        logger.info(f"Running evaluation with metrics: {request.metrics}")
        
        # Step 6: Run evaluation (use async version to avoid uvloop conflict)
        import asyncio
        result = await aevaluate(
            dataset=evaluation_dataset,
            metrics=selected_metrics,
            llm=evaluator_llm
        )
        
        # Step 7: Save results
        output_dir = os.path.join(request.output_folder, request.scenario_name, "04_evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.xlsx")
        
        # Create detailed results dataframe
        results_df = df.copy()
        
        # RAGAS 0.3.6 returns a Dataset-like object with scores
        # Convert to pandas DataFrame first
        try:
            import pandas as pd
            
            # Debug: Log result type and structure
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result attributes: {[x for x in dir(result) if not x.startswith('_')][:20]}")
            
            # Get the scores as a dictionary
            result_dict = result.to_pandas() if hasattr(result, 'to_pandas') else result
            logger.info(f"Result dict type: {type(result_dict)}")
            
            if isinstance(result_dict, pd.DataFrame):
                logger.info(f"Result DataFrame columns: {result_dict.columns.tolist()}")
                logger.info(f"Result DataFrame shape: {result_dict.shape}")
            
            # Add metric columns to results
            # RAGAS實際使用的列名映射
            ragas_column_mapping = {
                'answer_accuracy': 'nv_accuracy',
                'factual_correctness': 'factual_correctness(mode=f1)',
                'response_relevancy': 'answer_relevancy',
                'context_relevance': 'nv_context_relevance',
                'answer_correctness': 'answer_correctness',
                'answer_relevancy': 'answer_relevancy',
                'answer_similarity': 'answer_similarity',
                'context_precision': 'context_precision',
                'context_recall': 'context_recall',
                'faithfulness': 'faithfulness'
            }
            
            for metric_name in request.metrics:
                try:
                    # 首先嘗試使用映射的列名
                    ragas_column = ragas_column_mapping.get(metric_name, metric_name)
                    
                    if isinstance(result_dict, pd.DataFrame) and ragas_column in result_dict.columns:
                        results_df[metric_name] = result_dict[ragas_column]
                        logger.info(f"Added {metric_name} from DataFrame column '{ragas_column}'")
                    elif isinstance(result_dict, pd.DataFrame) and metric_name in result_dict.columns:
                        results_df[metric_name] = result_dict[metric_name]
                        logger.info(f"Added {metric_name} from DataFrame column '{metric_name}'")
                    elif hasattr(result, metric_name):
                        results_df[metric_name] = getattr(result, metric_name)
                        logger.info(f"Added {metric_name} from result attribute")
                    elif hasattr(result, '_scores_dict') and metric_name in result._scores_dict:
                        results_df[metric_name] = result._scores_dict[metric_name]
                        logger.info(f"Added {metric_name} from _scores_dict")
                    else:
                        logger.warning(f"Metric {metric_name} not found in result (tried column '{ragas_column}')")
                        results_df[metric_name] = None
                except Exception as e:
                    logger.warning(f"Could not add metric {metric_name} to results: {e}")
                    results_df[metric_name] = None
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            import traceback
            traceback.print_exc()
        
        results_df.to_excel(result_file, index=False)
        
        # Calculate average scores
        scores = {}
        for metric_name in request.metrics:
            try:
                # Try to get the metric value from result using column mapping
                metric_value = None
                ragas_column = ragas_column_mapping.get(metric_name, metric_name)
                
                if isinstance(result_dict, pd.DataFrame) and ragas_column in result_dict.columns:
                    metric_value = result_dict[ragas_column].mean()
                    logger.info(f"Calculated {metric_name} score from column '{ragas_column}': {metric_value}")
                elif isinstance(result_dict, pd.DataFrame) and metric_name in result_dict.columns:
                    metric_value = result_dict[metric_name].mean()
                    logger.info(f"Calculated {metric_name} score from column '{metric_name}': {metric_value}")
                elif hasattr(result, metric_name):
                    val = getattr(result, metric_name)
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        metric_value = sum(val) / len(val) if len(val) > 0 else 0
                    else:
                        metric_value = float(val)
                elif hasattr(result, '_scores_dict') and metric_name in result._scores_dict:
                    val = result._scores_dict[metric_name]
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        metric_value = sum(val) / len(val) if len(val) > 0 else 0
                    else:
                        metric_value = float(val)
                
                # Convert to float and handle NaN/Inf
                if metric_value is not None:
                    import math
                    float_val = float(metric_value)
                    # Replace NaN and Inf with 0.0
                    if math.isnan(float_val) or math.isinf(float_val):
                        scores[metric_name] = 0.0
                        logger.warning(f"Metric {metric_name} had NaN/Inf value, set to 0.0")
                    else:
                        scores[metric_name] = float_val
                else:
                    scores[metric_name] = 0.0
                    logger.warning(f"Could not find metric {metric_name} in result")
            except Exception as e:
                logger.warning(f"Could not calculate score for {metric_name}: {e}")
                scores[metric_name] = 0.0
        
        evaluation_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Results saved to {result_file}")
        logger.info(f"Scores: {scores}")
        
        # Ensure all scores are JSON-compliant
        import math
        clean_scores = {}
        for k, v in scores.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_scores[k] = 0.0
            else:
                clean_scores[k] = v
        
        return {
            "success": True,
            "scores": clean_scores,
            "total_tests": len(df),
            "evaluation_time": f"{evaluation_time:.2f}s",
            "result_file": result_file
        }
        
    except Exception as e:
        logger.error(f"Error running workflow evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_evaluation_result(file_path: str):
    """Download evaluation result file."""
    try:
        from fastapi.responses import FileResponse
        import os
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=os.path.basename(file_path)
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))