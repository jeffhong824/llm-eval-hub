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
    
    def __init__(
        self, 
        judge_model_type: JudgeModelType, 
        judge_model: str,
        conversation_llm_provider: str = "openai",
        conversation_llm_model: str = "gpt-4"
    ):
        self.judge_model_type = judge_model_type
        self.judge_model = judge_model
        self.multi_judge = MultiModelJudge()
        
        # LLM for generating user responses and judging goal achievement
        self.conversation_llm_provider = conversation_llm_provider
        self.conversation_llm_model = conversation_llm_model
        self.conversation_llm = self._create_conversation_llm()
    
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
        """Check if the task is complete based on conversation using LLM judge."""
        if not conversation or len(conversation) < 2:
            return False
        
        # Use LLM to judge if goal is achieved
        conversation_text = "\n\n".join([
            f"{'用戶' if turn.speaker == 'user' else 'Agent'}: {turn.message}"
            for turn in conversation
        ])
        
        prompt = f"""請判斷以下對話中，Agent 是否已經達成用戶的目標。

## 用戶角色和目標
角色背景: {test_case.user_persona}
任務情境: {test_case.task_context}
初始請求: {test_case.initial_request}
期望結果: {test_case.expected_outcome}
成功標準: {test_case.success_criteria}

## 對話記錄
{conversation_text}

## 判斷標準
請根據以下標準判斷任務是否完成：
1. Agent 是否提供了用戶期望的結果
2. 是否滿足成功標準中列出的要求
3. 用戶的核心問題是否得到解決
4. 對話是否達到了自然的結束點

請回答 "YES" 或 "NO"，並簡短說明理由。

格式：
```json
{{
  "goal_achieved": true/false,
  "confidence": 0.0-1.0,
  "reason": "簡短說明理由"
}}
```

只返回JSON，不要有其他文字。"""

        try:
            response = self.conversation_llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            goal_achieved = result.get("goal_achieved", False)
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"Goal achievement check: {goal_achieved} (confidence: {confidence})")
            logger.info(f"Reason: {result.get('reason', 'N/A')}")
            
            # Only return True if confidence is high enough
            return goal_achieved and confidence >= 0.7
            
        except Exception as e:
            logger.error(f"Error checking task completion: {e}")
            # Fallback to simple keyword check
            last_agent_message = conversation[-1].message.lower()
            completion_keywords = ['完成', '完成任務', '任務完成', 'done', 'completed', 'finished']
            return any(keyword in last_agent_message for keyword in completion_keywords)
    
    async def _generate_next_user_message(self, test_case: AgentTestCase, conversation: List[AgentConversationTurn]) -> str:
        """Generate next user message based on persona and conversation context using LLM."""
        # Build conversation context
        conversation_text = "\n\n".join([
            f"{'用戶' if turn.speaker == 'user' else 'Agent'}: {turn.message}"
            for turn in conversation[-6:]  # Use last 3 exchanges for context
        ])
        
        # Get the last agent response
        last_agent_message = conversation[-1].message if conversation else ""
        
        prompt = f"""你現在要扮演一個真實的用戶，根據你的角色特徵和當前對話情境，生成下一個自然的回應。

## 你的角色資訊
{json.dumps(test_case.user_persona, ensure_ascii=False, indent=2)}

## 任務情境
- 任務情境: {test_case.task_context}
- 你的目標: {test_case.expected_outcome}
- 初始請求: {test_case.initial_request}

## 當前對話記錄
{conversation_text}

## Agent 的最新回應
{last_agent_message}

## 你的任務
根據你的角色特徵（個性、溝通風格、需求、擔憂等），生成一個自然的下一句話。

考慮因素：
1. 你的溝通風格和語言特色
2. 你的需求是否得到滿足
3. 你還有什麼疑問或擔憂
4. 根據 Agent 的回應，你的下一步行動
5. 保持角色一致性
6. 如果目標已達成且滿意，可以表達感謝並結束對話
7. 如果還有疑問或需要更多信息，繼續追問

請生成一個簡短、自然的用戶回應（1-3句話）。

格式：
```json
{{
  "message": "你的回應內容",
  "intent": "clarification|follow_up|confirmation|new_question|satisfaction|concern",
  "reasoning": "為什麼這樣回應的簡短理由"
}}
```

只返回JSON，不要有其他文字。"""

        try:
            response = self.conversation_llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            message = result.get("message", "")
            intent = result.get("intent", "follow_up")
            
            logger.info(f"Generated next user message (intent: {intent}): {message[:100]}...")
            
            return message
            
        except Exception as e:
            logger.error(f"Error generating next user message: {e}")
            # Fallback to simple follow-up
            return "請繼續說明，我需要更多資訊。"
    
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


