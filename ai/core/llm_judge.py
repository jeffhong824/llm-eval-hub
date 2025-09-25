"""LLM Judge system for evaluating LLM responses."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from configs.settings import settings

logger = logging.getLogger(__name__)


class EvaluationCriteria(BaseModel):
    """Evaluation criteria for LLM judge."""
    
    criteria_name: str
    description: str
    evaluation_prompt: str
    scoring_scale: Dict[str, Union[int, float]] = Field(
        default={"min": 0, "max": 1, "step": 0.1}
    )
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class JudgeRequest(BaseModel):
    """Request for LLM judge evaluation."""
    
    question: str
    ground_truth: Optional[str] = None
    context: Optional[str] = None
    llm_response: str
    evaluation_criteria: List[EvaluationCriteria]
    judge_model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class JudgeResponse(BaseModel):
    """Response from LLM judge evaluation."""
    
    overall_score: float
    criteria_scores: Dict[str, float]
    reasoning: str
    confidence: float
    evaluation_details: Dict[str, Any]


@dataclass
class BatchJudgeResult:
    """Result of batch evaluation by LLM judge."""
    
    batch_id: str
    timestamp: datetime
    total_samples: int
    successful_evaluations: int
    failed_evaluations: int
    average_score: float
    score_distribution: Dict[str, int]
    evaluation_time: float
    results: List[JudgeResponse]
    errors: List[str]


class LLMJudge:
    """LLM Judge system for evaluating LLM responses."""
    
    def __init__(self, openai_api_key: str):
        """Initialize the LLM Judge."""
        self.openai_api_key = openai_api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Default evaluation criteria
        self.default_criteria = {
            "accuracy": EvaluationCriteria(
                criteria_name="accuracy",
                description="How accurate is the response compared to ground truth?",
                evaluation_prompt="""
                Evaluate the accuracy of the response compared to the ground truth.
                Consider factual correctness, completeness, and precision.
                Score from 0 (completely inaccurate) to 1 (perfectly accurate).
                """,
                weight=0.3
            ),
            "factual_correctness": EvaluationCriteria(
                criteria_name="factual_correctness",
                description="Are the facts presented in the response correct?",
                evaluation_prompt="""
                Evaluate the factual correctness of the response.
                Check for any false information, hallucinations, or incorrect claims.
                Score from 0 (many factual errors) to 1 (completely factually correct).
                """,
                weight=0.25
            ),
            "precision": EvaluationCriteria(
                criteria_name="precision",
                description="How precise and specific is the response?",
                evaluation_prompt="""
                Evaluate the precision of the response.
                Consider specificity, detail level, and relevance to the question.
                Score from 0 (very imprecise/vague) to 1 (highly precise and specific).
                """,
                weight=0.2
            ),
            "recall": EvaluationCriteria(
                criteria_name="recall",
                description="How well does the response cover the required information?",
                evaluation_prompt="""
                Evaluate the recall of the response.
                Check if all important aspects of the question are addressed.
                Score from 0 (misses most important points) to 1 (covers all important points).
                """,
                weight=0.15
            ),
            "f1": EvaluationCriteria(
                criteria_name="f1",
                description="Harmonic mean of precision and recall",
                evaluation_prompt="""
                Calculate the F1 score as the harmonic mean of precision and recall.
                This balances both precision and recall performance.
                Score from 0 (poor balance) to 1 (perfect balance).
                """,
                weight=0.1
            )
        }
    
    async def evaluate_single_response(
        self,
        request: JudgeRequest
    ) -> JudgeResponse:
        """
        Evaluate a single LLM response using the judge LLM.
        
        Args:
            request: JudgeRequest with evaluation details
            
        Returns:
            JudgeResponse with evaluation results
        """
        try:
            # Prepare evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(request)
            
            # Call judge LLM
            judge_response = await self._call_judge_llm(
                prompt=evaluation_prompt,
                model=request.judge_model,
                temperature=request.temperature
            )
            
            # Parse response
            parsed_response = self._parse_judge_response(judge_response, request)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise
    
    async def evaluate_batch(
        self,
        requests: List[JudgeRequest],
        batch_size: int = 10
    ) -> BatchJudgeResult:
        """
        Evaluate multiple responses in batch.
        
        Args:
            requests: List of JudgeRequest objects
            batch_size: Number of requests to process concurrently
            
        Returns:
            BatchJudgeResult with batch evaluation results
        """
        start_time = datetime.now()
        batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        results = []
        errors = []
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [
                self.evaluate_single_response(req) 
                for req in batch_requests
            ]
            
            # Execute batch
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    else:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                errors.append(str(e))
        
        # Calculate statistics
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        if results:
            scores = [r.overall_score for r in results]
            average_score = sum(scores) / len(scores)
            
            # Score distribution
            score_distribution = self._calculate_score_distribution(scores)
        else:
            average_score = 0.0
            score_distribution = {}
        
        return BatchJudgeResult(
            batch_id=batch_id,
            timestamp=start_time,
            total_samples=len(requests),
            successful_evaluations=len(results),
            failed_evaluations=len(errors),
            average_score=average_score,
            score_distribution=score_distribution,
            evaluation_time=evaluation_time,
            results=results,
            errors=errors
        )
    
    def _build_evaluation_prompt(self, request: JudgeRequest) -> str:
        """Build the evaluation prompt for the judge LLM."""
        
        prompt = f"""
        You are an expert evaluator tasked with assessing LLM responses. 
        Please evaluate the following response based on the specified criteria.

        Question: {request.question}
        
        Context: {request.context or "No context provided"}
        
        Ground Truth: {request.ground_truth or "No ground truth provided"}
        
        LLM Response: {request.llm_response}
        
        Evaluation Criteria:
        """
        
        for criteria in request.evaluation_criteria:
            prompt += f"""
        {criteria.criteria_name}: {criteria.description}
        {criteria.evaluation_prompt}
        Weight: {criteria.weight}
        """
        
        prompt += """
        
        Please provide your evaluation in the following JSON format:
        {
            "overall_score": <float between 0 and 1>,
            "criteria_scores": {
                "<criteria_name>": <float between 0 and 1>,
                ...
            },
            "reasoning": "<detailed explanation of your evaluation>",
            "confidence": <float between 0 and 1>,
            "evaluation_details": {
                "strengths": ["<strength1>", "<strength2>", ...],
                "weaknesses": ["<weakness1>", "<weakness2>", ...],
                "suggestions": ["<suggestion1>", "<suggestion2>", ...]
            }
        }
        
        Be thorough, objective, and provide constructive feedback.
        """
        
        return prompt
    
    async def _call_judge_llm(
        self, 
        prompt: str, 
        model: str, 
        temperature: float
    ) -> str:
        """Call the judge LLM API."""
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert evaluator of LLM responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"]
    
    def _parse_judge_response(
        self, 
        response: str, 
        request: JudgeRequest
    ) -> JudgeResponse:
        """Parse the judge LLM response."""
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            return JudgeResponse(
                overall_score=float(parsed_data.get("overall_score", 0.0)),
                criteria_scores=parsed_data.get("criteria_scores", {}),
                reasoning=parsed_data.get("reasoning", ""),
                confidence=float(parsed_data.get("confidence", 0.0)),
                evaluation_details=parsed_data.get("evaluation_details", {})
            )
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {str(e)}")
            # Return default response
            return JudgeResponse(
                overall_score=0.0,
                criteria_scores={},
                reasoning=f"Error parsing response: {str(e)}",
                confidence=0.0,
                evaluation_details={}
            )
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution."""
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution
    
    def get_default_criteria(self) -> Dict[str, EvaluationCriteria]:
        """Get default evaluation criteria."""
        return self.default_criteria.copy()
    
    def create_custom_criteria(
        self,
        name: str,
        description: str,
        evaluation_prompt: str,
        weight: float = 1.0
    ) -> EvaluationCriteria:
        """Create custom evaluation criteria."""
        return EvaluationCriteria(
            criteria_name=name,
            description=description,
            evaluation_prompt=evaluation_prompt,
            weight=weight
        )



