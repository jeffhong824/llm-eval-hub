"""Tests for the evaluator module."""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai.core.evaluator import LLMEvaluator, EvaluationResult
from ai.core.llm_judge import LLMJudge, JudgeRequest, EvaluationCriteria
from ai.testset.generator import TestsetGeneratorService, TestsetConfig, GeneratedTestset
from langchain.schema import Document


class TestLLMEvaluator:
    """Test cases for LLMEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing."""
        with patch('ai.core.evaluator.Client'):
            return LLMEvaluator()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "test.txt"}
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={"source": "test2.txt"}
            )
        ]
    
    @pytest.fixture
    def sample_testset_data(self):
        """Sample testset data for testing."""
        return pd.DataFrame({
            "question": ["What is ML?", "What is DL?"],
            "context": ["ML is a subset of AI", "DL uses neural networks"],
            "answer": ["ML is AI subset", "DL uses networks"],
            "ground_truth": ["ML is AI subset", "DL uses networks"]
        })
    
    @pytest.mark.asyncio
    async def test_generate_testset(self, evaluator, sample_documents):
        """Test testset generation."""
        with patch('ai.core.evaluator.TestsetGenerator') as mock_generator:
            mock_generator.return_value.generate.return_value = pd.DataFrame({
                "question": ["Test question"],
                "context": ["Test context"],
                "answer": ["Test answer"]
            })
            
            result = await evaluator.generate_testset(sample_documents, testset_size=5)
            
            assert isinstance(result, EvaluationResult)
            assert result.status in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_evaluate_rag_system(self, evaluator, sample_testset_data):
        """Test RAG system evaluation."""
        with patch('ai.core.evaluator.evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                "accuracy": 0.85,
                "factual_correctness": 0.90
            }
            
            result = await evaluator.evaluate_rag_system(
                testset_data=sample_testset_data,
                llm_endpoint="https://api.example.com",
                metrics=["accuracy", "factual_correctness"]
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.status in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_evaluate_agent_system(self, evaluator, sample_testset_data):
        """Test agent system evaluation."""
        with patch('ai.core.evaluator.evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                "accuracy": 0.80,
                "average_turn": 3.5
            }
            
            result = await evaluator.evaluate_agent_system(
                testset_data=sample_testset_data,
                agent_endpoint="https://api.example.com",
                metrics=["accuracy", "average_turn"]
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.status in ["success", "error"]


class TestLLMJudge:
    """Test cases for LLMJudge."""
    
    @pytest.fixture
    def judge(self):
        """Create judge instance for testing."""
        return LLMJudge("test-api-key")
    
    @pytest.fixture
    def sample_judge_request(self):
        """Sample judge request for testing."""
        criteria = EvaluationCriteria(
            criteria_name="accuracy",
            description="Test accuracy",
            evaluation_prompt="Test prompt",
            weight=1.0
        )
        
        return JudgeRequest(
            question="What is ML?",
            ground_truth="ML is AI subset",
            llm_response="ML is AI subset",
            evaluation_criteria=[criteria]
        )
    
    @pytest.mark.asyncio
    async def test_evaluate_single_response(self, judge, sample_judge_request):
        """Test single response evaluation."""
        with patch.object(judge, '_call_judge_llm') as mock_call:
            mock_call.return_value = '''
            {
                "overall_score": 0.9,
                "criteria_scores": {"accuracy": 0.9},
                "reasoning": "Good response",
                "confidence": 0.8,
                "evaluation_details": {"strengths": ["accurate"], "weaknesses": []}
            }
            '''
            
            result = await judge.evaluate_single_response(sample_judge_request)
            
            assert result.overall_score == 0.9
            assert result.confidence == 0.8
            assert "accuracy" in result.criteria_scores
    
    @pytest.mark.asyncio
    async def test_evaluate_batch(self, judge):
        """Test batch evaluation."""
        requests = [
            JudgeRequest(
                question="Q1",
                llm_response="A1",
                evaluation_criteria=list(judge.get_default_criteria().values())
            ),
            JudgeRequest(
                question="Q2", 
                llm_response="A2",
                evaluation_criteria=list(judge.get_default_criteria().values())
            )
        ]
        
        with patch.object(judge, 'evaluate_single_response') as mock_eval:
            mock_eval.return_value = Mock(
                overall_score=0.8,
                criteria_scores={"accuracy": 0.8},
                reasoning="Good",
                confidence=0.7,
                evaluation_details={}
            )
            
            result = await judge.evaluate_batch(requests)
            
            assert result.total_samples == 2
            assert result.successful_evaluations == 2
            assert result.failed_evaluations == 0
    
    def test_get_default_criteria(self, judge):
        """Test getting default criteria."""
        criteria = judge.get_default_criteria()
        
        assert "accuracy" in criteria
        assert "factual_correctness" in criteria
        assert "precision" in criteria
        assert "recall" in criteria
        assert "f1" in criteria
    
    def test_create_custom_criteria(self, judge):
        """Test creating custom criteria."""
        criteria = judge.create_custom_criteria(
            name="custom_test",
            description="Custom test criteria",
            evaluation_prompt="Test prompt",
            weight=0.5
        )
        
        assert criteria.criteria_name == "custom_test"
        assert criteria.weight == 0.5


class TestTestsetGeneratorService:
    """Test cases for TestsetGeneratorService."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return TestsetGeneratorService("test-api-key")
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Test document content",
                metadata={"source": "test.txt"}
            )
        ]
    
    @pytest.fixture
    def sample_config(self):
        """Sample testset config for testing."""
        return TestsetConfig(
            testset_size=5,
            single_hop_ratio=0.5,
            multi_hop_abstract_ratio=0.25,
            multi_hop_specific_ratio=0.25
        )
    
    @pytest.mark.asyncio
    async def test_generate_rag_testset(self, generator, sample_documents, sample_config):
        """Test RAG testset generation."""
        with patch('ai.testset.generator.TestsetGenerator') as mock_generator:
            mock_generator.return_value.generate.return_value = pd.DataFrame({
                "question": ["Test question"],
                "context": ["Test context"],
                "answer": ["Test answer"]
            })
            
            result = await generator.generate_rag_testset(sample_documents, sample_config)
            
            assert isinstance(result, GeneratedTestset)
            assert result.status in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_generate_from_user_scenarios(self, generator):
        """Test scenario-based testset generation."""
        scenarios = ["Test scenario 1", "Test scenario 2"]
        
        with patch.object(generator, '_generate_documents_from_scenarios') as mock_gen_docs:
            mock_gen_docs.return_value = [
                Document(page_content="Generated doc", metadata={})
            ]
            
            with patch('ai.testset.generator.TestsetGenerator') as mock_generator:
                mock_generator.return_value.generate.return_value = pd.DataFrame({
                    "question": ["Test question"],
                    "context": ["Test context"],
                    "answer": ["Test answer"]
                })
                
                result = await generator.generate_from_user_scenarios(scenarios)
                
                assert isinstance(result, GeneratedTestset)
                assert result.status in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_generate_agent_testset(self, generator):
        """Test agent testset generation."""
        scenarios = [
            {
                "id": "scenario_1",
                "description": "Test scenario",
                "type": "information_retrieval",
                "difficulty": "easy"
            }
        ]
        
        with patch('ai.testset.generator.TestsetGenerator') as mock_generator:
            mock_generator.return_value.generate.return_value = pd.DataFrame({
                "question": ["Test question"],
                "context": ["Test context"],
                "answer": ["Test answer"]
            })
            
            result = await generator.generate_agent_testset(scenarios)
            
            assert isinstance(result, GeneratedTestset)
            assert result.status in ["success", "error"]


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation workflow."""
        # This would test the complete workflow from testset generation to evaluation
        # For now, it's a placeholder
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        # This would test error handling
        # For now, it's a placeholder
        pass


if __name__ == "__main__":
    pytest.main([__file__])



