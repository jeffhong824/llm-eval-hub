"""
Agent Evaluation Module

Evaluates Agent systems using task scenarios from testset.
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
class AgentTestCase:
    """Single Agent test case."""
    task_id: str
    task_type: str
    user_persona: Dict[str, str]
    task_context: str
    initial_request: str
    expected_outcome: str
    success_criteria: str
    difficulty: str
    context_used: str


@dataclass
class AgentConversationTurn:
    """Single turn in agent conversation."""
    turn_number: int
    speaker: str  # "user" or "assistant"
    message: str
    timestamp: float


@dataclass
class AgentEvaluationResult:
    """Agent evaluation result for a single test case."""
    test_case: AgentTestCase
    conversation: List[AgentConversationTurn]
    final_response: str
    task_completion_score: float
    conversation_quality_score: float
    user_satisfaction_score: float
    goal_achievement_score: float
    average_turn_score: float
    success_rate: float
    judge_model_used: str
    evaluation_time: float
    total_turns: int


class AgentEvaluator:
    """Evaluates Agent systems using task scenarios."""
    
    def __init__(self, judge_model_type: JudgeModelType, judge_model: str):
        self.judge_model_type = judge_model_type
        self.judge_model = judge_model
        self.multi_judge = MultiModelJudge()
    
    def load_testset(self, excel_path: str) -> List[AgentTestCase]:
        """Load Agent testset from Excel file."""
        try:
            df = pd.read_excel(excel_path)
            test_cases = []
            
            for _, row in df.iterrows():
                # Parse user_persona from flattened columns
                user_persona = {
                    'role': row.get('user_role', ''),
                    'background': row.get('user_background', ''),
                    'communication_style': row.get('communication_style', ''),
                    'motivation': row.get('user_motivation', '')
                }
                
                test_case = AgentTestCase(
                    task_id=row.get('task_id', ''),
                    task_type=row.get('task_type', ''),
                    user_persona=user_persona,
                    task_context=row.get('task_context', ''),
                    initial_request=row.get('initial_request', ''),
                    expected_outcome=row.get('expected_outcome', ''),
                    success_criteria=row.get('success_criteria', ''),
                    difficulty=row.get('difficulty', 'medium'),
                    context_used=row.get('context_used', '')
                )
                test_cases.append(test_case)
            
            logger.info(f"Loaded {len(test_cases)} Agent test cases from {excel_path}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading Agent testset: {e}")
            raise
    
    async def call_llm_api(self, api_config: Dict[str, Any], message: str, conversation_id: str = None) -> str:
        """Call external LLM API for agent conversation."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = api_config.get('headers', {})
                url = api_config['url']
                
                # Prepare request data based on API type
                if 'workspace' in url:  # AnythingLLM API
                    data = {
                        "message": message,
                        "mode": "chat",
                        "sessionId": conversation_id or api_config.get('session_id', '1')
                    }
                elif 'conversations' in url:  # Custom conversation API
                    if conversation_id:
                        url = url.replace('conversations', f'conversations/{conversation_id}/chat')
                    data = {"text": message}
                else:  # Generic API
                    data = {"message": message}
                
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
    
    async def simulate_conversation(self, test_case: AgentTestCase, api_config: Dict[str, Any], max_turns: int = 30) -> List[AgentConversationTurn]:
        """Simulate a conversation with the agent."""
        import time
        
        conversation = []
        conversation_id = f"conv_{test_case.task_id}_{int(time.time())}"
        
        # Start with initial request
        current_message = test_case.initial_request
        turn_number = 1
        
        while turn_number <= max_turns:
            # User turn
            user_turn = AgentConversationTurn(
                turn_number=turn_number,
                speaker="user",
                message=current_message,
                timestamp=time.time()
            )
            conversation.append(user_turn)
            
            # Agent response
            agent_response = await self.call_llm_api(api_config, current_message, conversation_id)
            
            agent_turn = AgentConversationTurn(
                turn_number=turn_number,
                speaker="assistant",
                message=agent_response,
                timestamp=time.time()
            )
            conversation.append(agent_turn)
            
            # Check if task is complete (simple heuristic)
            if self._is_task_complete(test_case, conversation):
                break
            
            # Generate next user message based on persona and context
            current_message = await self._generate_next_user_message(test_case, conversation)
            
            turn_number += 1
        
        return conversation
    
    def _is_task_complete(self, test_case: AgentTestCase, conversation: List[AgentConversationTurn]) -> bool:
        """Check if the task is complete based on conversation."""
        if not conversation:
            return False
        
        # Simple heuristic: if agent mentions completion keywords
        last_agent_message = conversation[-1].message.lower()
        completion_keywords = ['完成', '完成任務', '任務完成', 'done', 'completed', 'finished']
        
        return any(keyword in last_agent_message for keyword in completion_keywords)
    
    async def _generate_next_user_message(self, test_case: AgentTestCase, conversation: List[AgentConversationTurn]) -> str:
        """Generate next user message based on persona and conversation context."""
        # This is a simplified version - in practice, you might want to use another LLM
        # to generate realistic follow-up messages based on the user persona
        
        persona = test_case.user_persona
        communication_style = persona.get('communication_style', '')
        
        # Simple follow-up messages based on communication style
        if '直接' in communication_style or 'direct' in communication_style.lower():
            follow_ups = [
                "我需要更多詳細信息",
                "這個回答不夠清楚",
                "請提供具體的步驟",
                "還有其他選擇嗎？"
            ]
        elif '好奇' in communication_style or 'curious' in communication_style.lower():
            follow_ups = [
                "這個很有趣，能告訴我更多嗎？",
                "為什麼會這樣？",
                "還有什麼其他相關的嗎？",
                "我想了解更多細節"
            ]
        else:
            follow_ups = [
                "謝謝，但我還有其他問題",
                "這個信息很有用",
                "請繼續說明",
                "我需要更多幫助"
            ]
        
        import random
        return random.choice(follow_ups)
    
    async def evaluate_single_case(self, test_case: AgentTestCase, api_config: Dict[str, Any], max_turns: int = 30) -> AgentEvaluationResult:
        """Evaluate a single Agent test case."""
        import time
        start_time = time.time()
        
        try:
            # Simulate conversation
            conversation = await self.simulate_conversation(test_case, api_config, max_turns)
            
            # Get final response
            final_response = conversation[-1].message if conversation else "No response"
            
            # Use judge model to evaluate
            evaluation_data = {
                "task_context": test_case.task_context,
                "expected_outcome": test_case.expected_outcome,
                "success_criteria": test_case.success_criteria,
                "conversation": [{"speaker": turn.speaker, "message": turn.message} for turn in conversation],
                "final_response": final_response
            }
            
            scores = await self.multi_judge.evaluate_batch(
                [evaluation_data],
                self.judge_model_type,
                self.judge_model
            )
            
            evaluation_time = time.time() - start_time
            
            # Calculate additional metrics
            task_completion_score = scores[0].get('task_completion', 0.0)
            conversation_quality_score = scores[0].get('conversation_quality', 0.0)
            user_satisfaction_score = scores[0].get('user_satisfaction', 0.0)
            goal_achievement_score = scores[0].get('goal_achievement', 0.0)
            
            # Calculate average turn score and success rate
            total_turns = len([turn for turn in conversation if turn.speaker == "user"])
            average_turn_score = conversation_quality_score / max(total_turns, 1)
            success_rate = 1.0 if task_completion_score > 0.7 else 0.0
            
            return AgentEvaluationResult(
                test_case=test_case,
                conversation=conversation,
                final_response=final_response,
                task_completion_score=task_completion_score,
                conversation_quality_score=conversation_quality_score,
                user_satisfaction_score=user_satisfaction_score,
                goal_achievement_score=goal_achievement_score,
                average_turn_score=average_turn_score,
                success_rate=success_rate,
                judge_model_used=f"{self.judge_model_type.value}:{self.judge_model}",
                evaluation_time=evaluation_time,
                total_turns=total_turns
            )
            
        except Exception as e:
            logger.error(f"Error evaluating agent test case: {e}")
            evaluation_time = time.time() - start_time
            
            return AgentEvaluationResult(
                test_case=test_case,
                conversation=[],
                final_response=f"Evaluation Error: {str(e)}",
                task_completion_score=0.0,
                conversation_quality_score=0.0,
                user_satisfaction_score=0.0,
                goal_achievement_score=0.0,
                average_turn_score=0.0,
                success_rate=0.0,
                judge_model_used=f"{self.judge_model_type.value}:{self.judge_model}",
                evaluation_time=evaluation_time,
                total_turns=0
            )
    
    async def evaluate_batch(self, test_cases: List[AgentTestCase], api_config: Dict[str, Any], max_turns: int = 30) -> List[AgentEvaluationResult]:
        """Evaluate multiple Agent test cases."""
        logger.info(f"Starting evaluation of {len(test_cases)} Agent test cases")
        
        # Process one by one to avoid overwhelming the API
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Processing Agent test case {i+1}/{len(test_cases)}: {test_case.task_id}")
            
            result = await self.evaluate_single_case(test_case, api_config, max_turns)
            results.append(result)
        
        logger.info(f"Completed evaluation of {len(results)} Agent test cases")
        return results
    
    def save_results(self, results: List[AgentEvaluationResult], output_path: str) -> Dict[str, Any]:
        """Save evaluation results to Excel file."""
        try:
            # Convert results to DataFrame
            data = []
            for result in results:
                # Flatten conversation data
                conversation_summary = f"{len(result.conversation)} turns"
                conversation_details = json.dumps([
                    {"turn": turn.turn_number, "speaker": turn.speaker, "message": turn.message}
                    for turn in result.conversation
                ])
                
                data.append({
                    'task_id': result.test_case.task_id,
                    'task_type': result.test_case.task_type,
                    'user_role': result.test_case.user_persona.get('role', ''),
                    'user_background': result.test_case.user_persona.get('background', ''),
                    'task_context': result.test_case.task_context,
                    'initial_request': result.test_case.initial_request,
                    'expected_outcome': result.test_case.expected_outcome,
                    'success_criteria': result.test_case.success_criteria,
                    'difficulty': result.test_case.difficulty,
                    'final_response': result.final_response,
                    'total_turns': result.total_turns,
                    'conversation_summary': conversation_summary,
                    'conversation_details': conversation_details,
                    'task_completion_score': result.task_completion_score,
                    'conversation_quality_score': result.conversation_quality_score,
                    'user_satisfaction_score': result.user_satisfaction_score,
                    'goal_achievement_score': result.goal_achievement_score,
                    'average_turn_score': result.average_turn_score,
                    'success_rate': result.success_rate,
                    'judge_model_used': result.judge_model_used,
                    'evaluation_time': result.evaluation_time
                })
            
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            
            # Calculate summary statistics
            summary = {
                'total_tests': len(results),
                'average_task_completion': df['task_completion_score'].mean(),
                'average_conversation_quality': df['conversation_quality_score'].mean(),
                'average_user_satisfaction': df['user_satisfaction_score'].mean(),
                'average_goal_achievement': df['goal_achievement_score'].mean(),
                'average_turn_score': df['average_turn_score'].mean(),
                'overall_success_rate': df['success_rate'].mean(),
                'average_turns_per_conversation': df['total_turns'].mean(),
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


