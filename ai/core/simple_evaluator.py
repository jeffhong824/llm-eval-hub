"""Simplified evaluation module using available RAGAS features."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    context_utilization,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result data structure."""
    
    evaluation_id: str
    timestamp: datetime
    testset_size: int
    metrics: Dict[str, float]
    evaluation_time: float
    status: str
    error_message: Optional[str] = None


class SimpleLLMEvaluator:
    """Simplified LLM evaluator using RAGAS."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.1
            )
        )
        
        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model="text-embedding-3-small"
            )
        )
        
        # Available metrics mapping
        self.metrics_mapping = {
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_relevancy": context_relevancy,
            "context_utilization": context_utilization,
        }
    
    async def evaluate_dataset(
        self,
        dataset: pd.DataFrame,
        metrics: List[str],
        system_type: str = "rag"
    ) -> EvaluationResult:
        """
        Evaluate a dataset using specified metrics.
        
        Args:
            dataset: DataFrame with columns ['question', 'ground_truth', 'answer', 'contexts']
            metrics: List of metric names to evaluate
            system_type: Type of system being evaluated
            
        Returns:
            EvaluationResult with evaluation scores
        """
        start_time = datetime.now()
        evaluation_id = f"eval_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting evaluation with {len(dataset)} samples")
            
            # Select metrics to use
            selected_metrics = []
            for metric_name in metrics:
                if metric_name in self.metrics_mapping:
                    selected_metrics.append(self.metrics_mapping[metric_name])
                else:
                    logger.warning(f"Metric {metric_name} not available, skipping")
            
            if not selected_metrics:
                raise ValueError("No valid metrics selected")
            
            # Run evaluation
            result = await evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Extract scores
            scores = {}
            for metric_name in metrics:
                if metric_name in self.metrics_mapping:
                    scores[metric_name] = result.get(metric_name, 0.0)
            
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                testset_size=len(dataset),
                metrics=scores,
                evaluation_time=evaluation_time,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                testset_size=len(dataset),
                metrics={},
                evaluation_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error_message=str(e)
            )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return list(self.metrics_mapping.keys())
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for testing."""
        return pd.DataFrame({
            'question': [
                'What is machine learning?',
                'How does neural network work?',
                'What are the benefits of AI?'
            ],
            'ground_truth': [
                'Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.',
                'Neural networks are computing systems inspired by biological neural networks.',
                'AI can automate tasks, improve efficiency, and enable new capabilities.'
            ],
            'answer': [
                'Machine learning is a method of data analysis that automates analytical model building.',
                'Neural networks process information through interconnected nodes that mimic neurons.',
                'AI provides automation, efficiency gains, and enables innovative solutions.'
            ],
            'contexts': [
                ['Machine learning algorithms build mathematical models based on training data.'],
                ['Neural networks consist of layers of interconnected nodes that process information.'],
                ['Artificial intelligence can automate repetitive tasks and improve decision-making.']
            ]
        })



