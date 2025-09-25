"""
RAG Evaluation Module

Evaluates RAG systems using QA pairs from testset.
"""

import logging
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from ai.core.multi_judge import MultiModelJudge, JudgeModelType

logger = logging.getLogger(__name__)


@dataclass
class RAGTestCase:
    """Single RAG test case."""
    question: str
    answer: str
    context: str
    difficulty: str
    question_type: str


@dataclass
class RAGEvaluationResult:
    """RAG evaluation result for a single test case."""
    test_case: RAGTestCase
    llm_response: str
    accuracy_score: float
    factual_correctness_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    response_relevancy_score: float
    faithfulness_score: float
    context_precision_score: float
    context_recall_score: float
    judge_model_used: str
    evaluation_time: float


class RAGEvaluator:
    """Evaluates RAG systems using test cases."""
    
    def __init__(self, judge_model_type: JudgeModelType, judge_model: str):
        self.judge_model_type = judge_model_type
        self.judge_model = judge_model
        self.multi_judge = MultiModelJudge()
    
    def load_testset(self, excel_path: str) -> List[RAGTestCase]:
        """Load RAG testset from Excel file."""
        try:
            df = pd.read_excel(excel_path)
            test_cases = []
            
            for _, row in df.iterrows():
                test_case = RAGTestCase(
                    question=row.get('question', ''),
                    answer=row.get('answer', ''),
                    context=row.get('context', ''),
                    difficulty=row.get('difficulty', 'medium'),
                    question_type=row.get('question_type', 'practical')
                )
                test_cases.append(test_case)
            
            logger.info(f"Loaded {len(test_cases)} RAG test cases from {excel_path}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading RAG testset: {e}")
            raise
    
    async def call_llm_api(self, api_config: Dict[str, Any], question: str, context: str = "") -> str:
        """Call external LLM API."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = api_config.get('headers', {})
                url = api_config['url']
                
                # Prepare request data based on API type
                if 'workspace' in url:  # AnythingLLM API
                    data = {
                        "message": question,
                        "mode": "chat",
                        "sessionId": api_config.get('session_id', '1')
                    }
                elif 'conversations' in url:  # Custom conversation API
                    data = {"text": question}
                else:  # Generic API
                    data = {"message": question}
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Extract response text based on API response format
                        if 'response' in result:
                            return result['response']
                        elif 'message' in result:
                            return result['message']
                        elif 'text' in result:
                            return result['text']
                        else:
                            return str(result)
                    else:
                        logger.error(f"API call failed with status {response.status}")
                        return f"API Error: {response.status}"
                        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"API Error: {str(e)}"
    
    async def evaluate_single_case(self, test_case: RAGTestCase, api_config: Dict[str, Any]) -> RAGEvaluationResult:
        """Evaluate a single RAG test case."""
        import time
        start_time = time.time()
        
        try:
            # Call LLM API
            llm_response = await self.call_llm_api(api_config, test_case.question, test_case.context)
            
            # Use judge model to evaluate
            evaluation_data = {
                "question": test_case.question,
                "answer": test_case.answer,
                "context": test_case.context,
                "response": llm_response
            }
            
            scores = await self.multi_judge.evaluate_batch(
                [evaluation_data],
                self.judge_model_type,
                self.judge_model
            )
            
            evaluation_time = time.time() - start_time
            
            return RAGEvaluationResult(
                test_case=test_case,
                llm_response=llm_response,
                accuracy_score=scores[0].get('accuracy', 0.0),
                factual_correctness_score=scores[0].get('factual_correctness', 0.0),
                precision_score=scores[0].get('precision', 0.0),
                recall_score=scores[0].get('recall', 0.0),
                f1_score=scores[0].get('f1', 0.0),
                response_relevancy_score=scores[0].get('response_relevancy', 0.0),
                faithfulness_score=scores[0].get('faithfulness', 0.0),
                context_precision_score=scores[0].get('context_precision', 0.0),
                context_recall_score=scores[0].get('context_recall', 0.0),
                judge_model_used=f"{self.judge_model_type.value}:{self.judge_model}",
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
            evaluation_time = time.time() - start_time
            
            return RAGEvaluationResult(
                test_case=test_case,
                llm_response=f"Evaluation Error: {str(e)}",
                accuracy_score=0.0,
                factual_correctness_score=0.0,
                precision_score=0.0,
                recall_score=0.0,
                f1_score=0.0,
                response_relevancy_score=0.0,
                faithfulness_score=0.0,
                context_precision_score=0.0,
                context_recall_score=0.0,
                judge_model_used=f"{self.judge_model_type.value}:{self.judge_model}",
                evaluation_time=evaluation_time
            )
    
    async def evaluate_batch(self, test_cases: List[RAGTestCase], api_config: Dict[str, Any]) -> List[RAGEvaluationResult]:
        """Evaluate multiple RAG test cases."""
        logger.info(f"Starting evaluation of {len(test_cases)} RAG test cases")
        
        # Process in batches to avoid overwhelming the API
        batch_size = 5
        results = []
        
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_cases) + batch_size - 1)//batch_size}")
            
            batch_tasks = [self.evaluate_single_case(test_case, api_config) for test_case in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation error: {result}")
                else:
                    results.append(result)
        
        logger.info(f"Completed evaluation of {len(results)} RAG test cases")
        return results
    
    def save_results(self, results: List[RAGEvaluationResult], output_path: str) -> Dict[str, Any]:
        """Save evaluation results to Excel file."""
        try:
            # Convert results to DataFrame
            data = []
            for result in results:
                data.append({
                    'question': result.test_case.question,
                    'expected_answer': result.test_case.answer,
                    'context': result.test_case.context,
                    'llm_response': result.llm_response,
                    'difficulty': result.test_case.difficulty,
                    'question_type': result.test_case.question_type,
                    'accuracy_score': result.accuracy_score,
                    'factual_correctness_score': result.factual_correctness_score,
                    'precision_score': result.precision_score,
                    'recall_score': result.recall_score,
                    'f1_score': result.f1_score,
                    'response_relevancy_score': result.response_relevancy_score,
                    'faithfulness_score': result.faithfulness_score,
                    'context_precision_score': result.context_precision_score,
                    'context_recall_score': result.context_recall_score,
                    'judge_model_used': result.judge_model_used,
                    'evaluation_time': result.evaluation_time
                })
            
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            
            # Calculate summary statistics
            summary = {
                'total_tests': len(results),
                'average_accuracy': df['accuracy_score'].mean(),
                'average_factual_correctness': df['factual_correctness_score'].mean(),
                'average_precision': df['precision_score'].mean(),
                'average_recall': df['recall_score'].mean(),
                'average_f1': df['f1_score'].mean(),
                'average_response_relevancy': df['response_relevancy_score'].mean(),
                'average_faithfulness': df['faithfulness_score'].mean(),
                'average_context_precision': df['context_precision_score'].mean(),
                'average_context_recall': df['context_recall_score'].mean(),
                'total_evaluation_time': df['evaluation_time'].sum(),
                'judge_model_used': results[0].judge_model_used if results else 'N/A'
            }
            
            logger.info(f"Saved evaluation results to {output_path}")
            return {
                'success': True,
                'output_path': output_path,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return {
                'success': False,
                'error': str(e)
            }


