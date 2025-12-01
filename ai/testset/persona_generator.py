"""
Persona Generator Module

Automatically generates detailed user personas based on use case scenarios.
This module creates rich, diverse user profiles that represent different potential users
of a system, with comprehensive demographic, behavioral, and contextual details.
"""

# Import pydantic patch FIRST, before any langchain imports
import ai.core.pydantic_patch  # noqa: F401

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from configs.settings import settings

logger = logging.getLogger(__name__)


class PersonaGenerator:
    """Generates detailed user personas based on scenario descriptions."""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        """
        Initialize persona generator.
        
        Args:
            model_provider: LLM provider (openai, gemini, ollama)
            model_name: Specific model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create LLM instance based on provider and model."""
        if self.model_provider == "openai":
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model=self.model_name,
                temperature=0.8  # Higher temperature for more creative personas
            )
        elif self.model_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.8
            )
        elif self.model_provider == "ollama":
            return ChatOpenAI(
                api_key="ollama",
                model=self.model_name,
                base_url=settings.ollama_base_url,
                temperature=0.8
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    async def generate_personas(
        self,
        scenario_description: str,
        num_personas: int = 10,
        language: str = "繁體中文"
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed personas for a given scenario.
        
        Args:
            scenario_description: Description of the use case/scenario
            num_personas: Number of personas to generate
            language: Language for persona descriptions
            
        Returns:
            List of detailed persona dictionaries
        """
        logger.info(f"Generating {num_personas} personas for scenario: {scenario_description[:100]}...")
        logger.info(f"Using model: {self.model_provider}/{self.model_name}")
        
        prompt = f"""作為一個用戶研究專家，請根據以下使用情境生成{num_personas}個通用且簡潔的用戶角色檔案(personas)。

使用情境：
{scenario_description}

請生成多樣化的用戶角色，每個角色都應該是獨特的、真實的、具體的，但保持簡潔。

對於每個角色，請提供以下資訊：

1. **基本資訊**：
   - 姓名、年齡、性別、職業、教育程度、收入水平、居住地

2. **生活狀態**：
   - 婚姻狀況、家庭成員、住房狀況、通勤方式

3. **需求與痛點**：
   - 當前痛點（2-3個）、核心需求

4. **溝通風格**：
   - 表達清晰度

請以JSON格式返回，確保每個角色都是完整且獨特的。保持回應簡潔。

輸出格式：
```json
[
  {{
    "persona_id": "persona_001",
    "persona_name": "具體姓名",
    "basic_info": {{
      "age": 具體數字,
      "gender": "性別",
      "occupation": "具體職位和公司類型",
      "education": "教育程度",
      "income_level": "收入範圍",
      "location": "具體城市/地區"
    }},
    "life_status": {{
      "marital_status": "婚姻狀況",
      "family_members": "家庭成員詳情",
      "housing": "住房詳情",
      "commute": "通勤詳情"
    }},
    "needs_and_goals": {{
      "pain_points": ["痛點1", "痛點2", "痛點3"],
      "core_needs": ["需求1（最重要）", "需求2", "需求3"]
    }},
    "communication_style": {{
      "clarity_level": "表達清晰度（例如：簡潔直接、詳細完整、口語化等）"
    }}
  }}
]
```

請確保：
1. 每個角色都是真實可信的完整人物
2. 角色之間有足夠的多樣性（不同年齡段、職業、需求等）
3. 所有欄位都填寫完整，不要有空白或籠統的描述
4. 使用{language}語言

現在請生成{num_personas}個詳細的用戶角色。只返回JSON，不要有其他文字。"""

        try:
            # Use async invoke to avoid blocking the event loop
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            personas = json.loads(content)
            
            if not isinstance(personas, list):
                raise ValueError("Response is not a list of personas")
            
            # Validate personas and normalize field names
            for i, persona in enumerate(personas):
                if not persona.get("persona_id"):
                    persona["persona_id"] = f"persona_{i+1:03d}"
                if not persona.get("persona_name"):
                    raise ValueError(f"Persona {i+1} missing persona_name")
                
                # Normalize: convert needs_and_pain_points to needs_and_goals for consistency
                if "needs_and_pain_points" in persona and "needs_and_goals" not in persona:
                    persona["needs_and_goals"] = persona.pop("needs_and_pain_points")
            
            logger.info(f"Successfully generated {len(personas)} personas (requested: {num_personas})")
            if len(personas) != num_personas:
                logger.warning(f"Generated {len(personas)} personas but requested {num_personas}")
            return personas
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content length: {len(content)}")
            logger.error(f"Response content preview: {content[:500]}...")
            
            # Try to fix incomplete JSON
            try:
                fixed_content = self._fix_incomplete_json(content)
                personas = json.loads(fixed_content)
                logger.info("Successfully fixed and parsed JSON")
                
                if not isinstance(personas, list):
                    raise ValueError("Fixed response is not a list of personas")
                
                # Validate personas and normalize field names
                for i, persona in enumerate(personas):
                    if not persona.get("persona_id"):
                        persona["persona_id"] = f"persona_{i+1:03d}"
                    if not persona.get("persona_name"):
                        raise ValueError(f"Persona {i+1} missing persona_name")
                    
                    # Normalize: convert needs_and_pain_points to needs_and_goals for consistency
                    if "needs_and_pain_points" in persona and "needs_and_goals" not in persona:
                        persona["needs_and_goals"] = persona.pop("needs_and_pain_points")
                
                logger.info(f"Successfully generated {len(personas)} personas after fixing (requested: {num_personas})")
                if len(personas) != num_personas:
                    logger.warning(f"After fixing: Generated {len(personas)} personas but requested {num_personas}")
                return personas
                
            except Exception as fix_error:
                logger.error(f"Failed to fix JSON: {fix_error}")
                
                # Last resort: try generating fewer personas
                if num_personas > 3:
                    logger.warning(f"JSON parsing failed, retrying with fewer personas ({num_personas//2} instead of {num_personas})...")
                    return self.generate_personas(scenario_description, num_personas//2, language)
                else:
                    logger.error("Even with minimal personas, generation failed")
                    raise
        except Exception as e:
            logger.error(f"Error generating personas: {e}")
            raise
    
    def _fix_incomplete_json(self, content: str) -> str:
        """Try to fix incomplete JSON by finding the last complete persona."""
        logger.info("Attempting to fix incomplete JSON...")
        
        # First, try to fix common JSON issues
        fixed_content = self._fix_common_json_issues(content)
        
        # Try to parse the fixed content
        try:
            json.loads(fixed_content)
            logger.info("Successfully fixed JSON with common fixes")
            return fixed_content
        except json.JSONDecodeError:
            pass
        
        # If that doesn't work, try to find the last complete persona
        lines = fixed_content.split('\n')
        fixed_lines = []
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # Simple brace counting
            for char in line:
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        # If we've closed a persona object, check if it's complete
                        if brace_count == 1:  # Back to array level
                            # Try to parse what we have so far
                            test_content = '\n'.join(fixed_lines)
                            if test_content.strip().endswith(','):
                                test_content = test_content.rstrip(',')
                            test_content += '\n]'
                            
                            try:
                                json.loads(test_content)
                                logger.info("Found complete JSON structure")
                                return test_content
                            except json.JSONDecodeError:
                                continue
        
        # If we get here, try to close the JSON properly
        final_content = fixed_content.strip()
        if final_content.endswith(','):
            final_content = final_content.rstrip(',')
        if not final_content.endswith(']'):
            final_content += '\n]'
            
        return final_content
    
    def _fix_common_json_issues(self, content: str) -> str:
        """Fix common JSON syntax issues."""
        logger.info("Applying common JSON fixes...")
        
        # Remove any trailing commas before closing braces/brackets
        import re
        
        # Fix trailing commas before closing braces
        content = re.sub(r',(\s*})', r'\1', content)
        # Fix trailing commas before closing brackets
        content = re.sub(r',(\s*\])', r'\1', content)
        
        # Fix missing commas between objects in array
        # Look for } followed by { without a comma
        content = re.sub(r'}(\s*){', r'},\1{', content)
        
        # Fix missing quotes around keys (basic cases)
        # This is more complex and might break things, so we'll be conservative
        
        # Ensure proper array closing
        if not content.strip().endswith(']'):
            content = content.strip()
            if content.endswith(','):
                content = content.rstrip(',')
            content += '\n]'
        
        return content
    
    def save_personas_to_files(
        self,
        personas: List[Dict[str, Any]],
        output_folder: str,
        scenario_name: str = "scenario"
    ) -> Dict[str, Any]:
        """
        Save personas to JSON file only (simplified format).
        
        Args:
            personas: List of persona dictionaries
            output_folder: Output directory path
            scenario_name: Name of the scenario for file naming
            
        Returns:
            Dictionary with paths and metadata
        """
        try:
            # Create output folder structure
            output_path = Path(output_folder)
            personas_folder = output_path / "01_personas"
            # Thread-safe directory creation
            max_retries = 3
            for retry in range(max_retries):
                try:
                    personas_folder.mkdir(parents=True, exist_ok=True)
                    if personas_folder.exists():
                        break
                    if retry < max_retries - 1:
                        import time
                        time.sleep(0.1 * (retry + 1))
                except (PermissionError, OSError) as e:
                    if retry == max_retries - 1:
                        raise
                    import time
                    time.sleep(0.1 * (retry + 1))
            
            # Save full JSON (only format we keep) using atomic write
            json_file = personas_folder / f"{scenario_name}_personas_full.json"
            temp_file = json_file.with_suffix('.json.tmp')
            try:
                # Write to temp file first
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(personas, f, ensure_ascii=False, indent=2)
                # Atomic rename
                if json_file.exists():
                    json_file.unlink()
                temp_file.replace(json_file)
            except Exception as write_error:
                # Clean up temp file on error
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                raise write_error
            
            logger.info(f"Saved {len(personas)} personas to JSON: {json_file}")
            
            return {
                "success": True,
                "personas_folder": str(personas_folder),
                "json_file": str(json_file),
                "total_personas": len(personas)
            }
            
        except Exception as e:
            logger.error(f"Error saving personas: {e}")
            return {
                "success": False,
                "error": str(e)
            }

