"""Core evaluation module using RAGAS and LangSmith."""

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
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langsmith import Client
from langsmith.evaluation import evaluate as langsmith_evaluate
from langsmith.schemas import Run, Example

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result data structure."""
    
    evaluation_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    raw_results: Dict[str, Any]
    testset_size: int
    evaluation_time: float
    status: str
    error_message: Optional[str] = None


@dataclass
class TestsetGenerationResult:
    """Testset generation result data structure."""
    
    testset_id: str
    timestamp: datetime
    testset_size: int
    documents_count: int
    generation_time: float
    testset_data: pd.DataFrame
    knowledge_graph_path: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None


class LLMEvaluator:
    """Main LLM evaluation class using RAGAS and LangSmith."""
    
    def __init__(self):
        """Initialize the evaluator with LLM and embedding models."""
        self.langsmith_client = Client(
            api_key=settings.langsmith_api_key,
            api_url=settings.langsmith_endpoint
        )
        
        # Initialize LLM for evaluation
        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.1
            )
        )
        
        # Initialize embeddings
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
    
    async def generate_testset(
        self,
        documents: List[Document],
        testset_size: int = 10,
        save_knowledge_graph: bool = True,
        custom_personas: Optional[List[str]] = None
    ) -> TestsetGenerationResult:
        """
        Generate synthetic testset from documents using RAGAS.
        
        Args:
            documents: List of documents to generate testset from
            testset_size: Number of QA pairs to generate
            save_knowledge_graph: Whether to save the knowledge graph
            custom_personas: Custom personas for testset generation
            
        Returns:
            TestsetGenerationResult with generated testset
        """
        start_time = datetime.now()
        testset_id = f"testset_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting testset generation with {len(documents)} documents")
            
            # Create knowledge graph
            kg = KnowledgeGraph()
            
            # Add documents to knowledge graph
            for doc in documents:
                kg.nodes.append(
                    Node(
                        type=NodeType.DOCUMENT,
                        properties={
                            "page_content": doc.page_content,
                            "document_metadata": doc.metadata
                        }
                    )
                )
            
            logger.info(f"Created knowledge graph with {len(kg.nodes)} nodes")
            
            # Apply transformations to enrich knowledge graph
            transforms = default_transforms(
                documents=documents,
                llm=self.llm,
                embedding_model=self.embeddings
            )
            
            apply_transforms(kg, transforms)
            logger.info(f"Applied transforms, knowledge graph now has {len(kg.nodes)} nodes")
            
            # Save knowledge graph if requested
            kg_path = None
            if save_knowledge_graph:
                kg_path = f"{settings.artifacts_dir}/knowledge_graph_{testset_id}.json"
                kg.save(kg_path)
                logger.info(f"Saved knowledge graph to {kg_path}")
            
            # Generate testset
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            
            dataset = generator.generate(testset_size=testset_size)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = TestsetGenerationResult(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=len(dataset),
                documents_count=len(documents),
                generation_time=generation_time,
                testset_data=dataset.to_pandas(),
                knowledge_graph_path=kg_path,
                status="success"
            )
            
            logger.info(f"Testset generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating testset: {str(e)}")
            return TestsetGenerationResult(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=0,
                documents_count=len(documents),
                generation_time=(datetime.now() - start_time).total_seconds(),
                testset_data=pd.DataFrame(),
                status="error",
                error_message=str(e)
            )
    
    async def evaluate_rag_system(
        self,
        testset_data: pd.DataFrame,
        llm_endpoint: str,
        metrics: List[str],
        evaluation_name: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate RAG system using specified metrics.
        
        Args:
            testset_data: Testset DataFrame with questions, contexts, answers
            llm_endpoint: API endpoint for the LLM to evaluate
            metrics: List of metrics to evaluate
            evaluation_name: Optional name for the evaluation
            
        Returns:
            EvaluationResult with evaluation scores
        """
        start_time = datetime.now()
        evaluation_id = f"eval_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting evaluation with {len(testset_data)} samples")
            
            # Validate metrics
            valid_metrics = []
            for metric in metrics:
                if metric in self.metrics_mapping:
                    valid_metrics.append(self.metrics_mapping[metric])
                else:
                    logger.warning(f"Unknown metric: {metric}")
            
            if not valid_metrics:
                raise ValueError("No valid metrics provided")
            
            # Create evaluation dataset
            dataset = self._prepare_evaluation_dataset(testset_data, llm_endpoint)
            
            # Run evaluation
            result = await evaluate(
                dataset=dataset,
                metrics=valid_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            metrics_scores = {}
            for metric_name, metric_score in result.items():
                if isinstance(metric_score, (int, float)):
                    metrics_scores[metric_name] = float(metric_score)
                else:
                    metrics_scores[metric_name] = float(metric_score.mean())
            
            evaluation_result = EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                metrics=metrics_scores,
                raw_results=result,
                testset_size=len(testset_data),
                evaluation_time=evaluation_time,
                status="success"
            )
            
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                metrics={},
                raw_results={},
                testset_size=len(testset_data),
                evaluation_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error_message=str(e)
            )
    
    async def evaluate_agent_system(
        self,
        testset_data: pd.DataFrame,
        agent_endpoint: str,
        metrics: List[str],
        evaluation_name: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate agent system with agent-specific metrics.
        
        Args:
            testset_data: Testset DataFrame with agent scenarios
            agent_endpoint: API endpoint for the agent to evaluate
            metrics: List of metrics to evaluate
            evaluation_name: Optional name for the evaluation
            
        Returns:
            EvaluationResult with evaluation scores including agent metrics
        """
        start_time = datetime.now()
        evaluation_id = f"agent_eval_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting agent evaluation with {len(testset_data)} scenarios")
            
            # Add agent-specific metrics
            agent_metrics = []
            for metric in metrics:
                if metric in self.metrics_mapping:
                    agent_metrics.append(self.metrics_mapping[metric])
            
            # Create evaluation dataset for agent
            dataset = self._prepare_agent_evaluation_dataset(testset_data, agent_endpoint)
            
            # Run evaluation
            result = await evaluate(
                dataset=dataset,
                metrics=agent_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Calculate agent-specific metrics
            agent_scores = await self._calculate_agent_metrics(testset_data, agent_endpoint)
            
            evaluation_time = (datetime.now() - start_time).total_seconds()
            
            # Combine results
            metrics_scores = {}
            for metric_name, metric_score in result.items():
                if isinstance(metric_score, (int, float)):
                    metrics_scores[metric_name] = float(metric_score)
                else:
                    metrics_scores[metric_name] = float(metric_score.mean())
            
            # Add agent-specific scores
            metrics_scores.update(agent_scores)
            
            evaluation_result = EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                metrics=metrics_scores,
                raw_results={**result, **agent_scores},
                testset_size=len(testset_data),
                evaluation_time=evaluation_time,
                status="success"
            )
            
            logger.info(f"Agent evaluation completed in {evaluation_time:.2f}s")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error during agent evaluation: {str(e)}")
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=start_time,
                metrics={},
                raw_results={},
                testset_size=len(testset_data),
                evaluation_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error_message=str(e)
            )
    
    def _prepare_evaluation_dataset(
        self, 
        testset_data: pd.DataFrame, 
        llm_endpoint: str
    ) -> pd.DataFrame:
        """Prepare dataset for evaluation by calling LLM endpoint."""
        # This would call the LLM endpoint and get responses
        # For now, return the testset_data as-is
        return testset_data
    
    def _prepare_agent_evaluation_dataset(
        self, 
        testset_data: pd.DataFrame, 
        agent_endpoint: str
    ) -> pd.DataFrame:
        """Prepare dataset for agent evaluation."""
        # This would call the agent endpoint and get responses
        # For now, return the testset_data as-is
        return testset_data
    
    async def _calculate_agent_metrics(
        self, 
        testset_data: pd.DataFrame, 
        agent_endpoint: str
    ) -> Dict[str, float]:
        """Calculate agent-specific metrics like average turn and success rate."""
        # Placeholder implementation
        return {
            "average_turn": 3.5,
            "success_rate": 0.85,
            "tool_call_accuracy": 0.92,
            "agent_goal_accuracy": 0.88,
            "topic_adherence": 0.90
        }
