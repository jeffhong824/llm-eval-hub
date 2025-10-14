"""
Persona Generator Module

Automatically generates detailed user personas based on use case scenarios.
This module creates rich, diverse user profiles that represent different potential users
of a system, with comprehensive demographic, behavioral, and contextual details.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
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
    
    def generate_personas(
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

請生成多樣化的用戶角色，每個角色都應該是獨特的、真實的、具體的，但不要過於複雜。

對於每個角色，請提供以下資訊（保持簡潔但完整）：

1. **基本資訊**：
   - 姓名、年齡、性別、職業、教育程度、收入水平、居住地

2. **生活狀態**：
   - 婚姻狀況、家庭成員、住房狀況、通勤方式

3. **需求與痛點**：
   - 當前痛點（2-3個）、核心需求

4. **行為模式**：
   - 日常作息、科技使用習慣

5. **心理特徵**：
   - 對該情境的態度

6. **場景特定需求**：
   - 根據情境的具體需求和偏好（這是最重要的部分）

7. **溝通風格**：
   - 問問題方式、語言特色

請以JSON格式返回，確保每個角色都是完整且獨特的。保持回應簡潔，避免過度詳細的描述。

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
    "needs_and_pain_points": {{
      "pain_points": ["痛點1", "痛點2", "痛點3"],
      "core_needs": ["需求1（最重要）", "需求2", "需求3"]
    }},
    "behavior_patterns": {{
      "daily_routine": "日常作息描述",
      "tech_usage": "科技產品使用習慣"
    }},
    "psychology": {{
      "attitude": "對該情境的態度和看法"
    }},
    "scenario_specific": {{
      // 根據具體情境添加專屬欄位和需求
      // 這是每個角色最重要的部分，要詳細描述
    }},
    "communication_style": {{
      "questioning_approach": "問問題的方式",
      "language_characteristics": "語言特色和表達方式"
    }}
  }}
]
```

請確保：
1. 每個角色都是真實可信的完整人物
2. 角色之間有足夠的多樣性（不同年齡段、職業、需求等）
3. 所有欄位都填寫完整，不要有空白或籠統的描述
4. scenario_specific 部分要根據具體情境添加相關欄位
5. 使用{language}語言

現在請生成{num_personas}個詳細的用戶角色。只返回JSON，不要有其他文字。"""

        try:
            response = self.llm.invoke(prompt)
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
            
            # Validate personas
            for i, persona in enumerate(personas):
                if not persona.get("persona_id"):
                    persona["persona_id"] = f"persona_{i+1:03d}"
                if not persona.get("persona_name"):
                    raise ValueError(f"Persona {i+1} missing persona_name")
            
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
                
                # Validate personas
                for i, persona in enumerate(personas):
                    if not persona.get("persona_id"):
                        persona["persona_id"] = f"persona_{i+1:03d}"
                    if not persona.get("persona_name"):
                        raise ValueError(f"Persona {i+1} missing persona_name")
                
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
        Save personas to individual files and summary Excel.
        
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
            personas_folder.mkdir(parents=True, exist_ok=True)
            
            # Save individual persona files
            persona_files = []
            for persona in personas:
                persona_id = persona.get("persona_id", "unknown")
                persona_name = persona.get("persona_name", "Unknown")
                
                # Create markdown file for each persona
                md_content = self._format_persona_as_markdown(persona)
                md_file = personas_folder / f"{persona_id}_{persona_name}.md"
                
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                persona_files.append(str(md_file))
                logger.info(f"Saved persona file: {md_file}")
            
            # Save summary Excel
            excel_file = personas_folder / f"{scenario_name}_personas_summary.xlsx"
            self._save_personas_excel(personas, excel_file)
            
            # Save full JSON
            json_file = personas_folder / f"{scenario_name}_personas_full.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(personas, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(personas)} persona files to {personas_folder}")
            
            return {
                "success": True,
                "personas_folder": str(personas_folder),
                "persona_files": persona_files,
                "excel_file": str(excel_file),
                "json_file": str(json_file),
                "total_personas": len(personas)
            }
            
        except Exception as e:
            logger.error(f"Error saving personas: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_persona_as_markdown(self, persona: Dict[str, Any]) -> str:
        """Format persona as a readable markdown document."""
        md = f"""# {persona.get('persona_name', 'Unknown Persona')}

## 角色ID
{persona.get('persona_id', 'N/A')}

## 基本資訊
- **年齡**: {persona.get('basic_info', {}).get('age', 'N/A')}
- **性別**: {persona.get('basic_info', {}).get('gender', 'N/A')}
- **職業**: {persona.get('basic_info', {}).get('occupation', 'N/A')}
- **教育程度**: {persona.get('basic_info', {}).get('education', 'N/A')}
- **收入水平**: {persona.get('basic_info', {}).get('income_level', 'N/A')}
- **居住地**: {persona.get('basic_info', {}).get('location', 'N/A')}

## 生活狀態
- **婚姻狀況**: {persona.get('life_status', {}).get('marital_status', 'N/A')}
- **家庭成員**: {persona.get('life_status', {}).get('family_members', 'N/A')}
- **寵物**: {persona.get('life_status', {}).get('pets', 'N/A')}
- **住房**: {persona.get('life_status', {}).get('housing', 'N/A')}
- **通勤**: {persona.get('life_status', {}).get('commute', 'N/A')}
- **工作地點**: {persona.get('life_status', {}).get('work_location', 'N/A')}

## 需求與目標

### 痛點
"""
        for pain_point in persona.get('needs_and_goals', {}).get('pain_points', []):
            md += f"- {pain_point}\n"
        
        md += "\n### 核心需求\n"
        for need in persona.get('needs_and_goals', {}).get('core_needs', []):
            md += f"- {need}\n"
        
        md += "\n### 短期目標\n"
        for goal in persona.get('needs_and_goals', {}).get('short_term_goals', []):
            md += f"- {goal}\n"
        
        md += "\n### 長期目標\n"
        for goal in persona.get('needs_and_goals', {}).get('long_term_goals', []):
            md += f"- {goal}\n"
        
        md += f"\n### 預算範圍\n{persona.get('needs_and_goals', {}).get('budget_range', 'N/A')}\n"
        
        md += "\n### 決策標準\n"
        for criterion in persona.get('needs_and_goals', {}).get('decision_criteria', []):
            md += f"- {criterion}\n"
        
        md += f"""
## 行為模式
- **日常作息**: {persona.get('behavior_patterns', {}).get('daily_routine', 'N/A')}
- **科技使用**: {persona.get('behavior_patterns', {}).get('tech_usage', 'N/A')}
- **資訊搜尋**: {persona.get('behavior_patterns', {}).get('info_seeking', 'N/A')}
- **決策風格**: {persona.get('behavior_patterns', {}).get('decision_style', 'N/A')}
- **社交媒體**: {persona.get('behavior_patterns', {}).get('social_media', 'N/A')}
- **溝通偏好**: {persona.get('behavior_patterns', {}).get('communication_preference', 'N/A')}

## 心理特徵

### 個性特點
"""
        for trait in persona.get('psychology', {}).get('personality_traits', []):
            md += f"- {trait}\n"
        
        md += "\n### 價值觀\n"
        for value in persona.get('psychology', {}).get('values', []):
            md += f"- {value}\n"
        
        md += f"""
- **態度**: {persona.get('psychology', {}).get('attitude', 'N/A')}
- **擔憂**: {', '.join(persona.get('psychology', {}).get('anxieties', []))}
- **動機**: {persona.get('psychology', {}).get('motivation', 'N/A')}
- **信任建立**: {persona.get('psychology', {}).get('trust_building', 'N/A')}

## 情境專屬需求
"""
        scenario_specific = persona.get('scenario_specific', {})
        for key, value in scenario_specific.items():
            md += f"- **{key}**: {value}\n"
        
        md += f"""
## 溝通風格
- **提問方式**: {persona.get('communication_style', {}).get('questioning_approach', 'N/A')}
- **語言特色**: {persona.get('communication_style', {}).get('language_characteristics', 'N/A')}
- **常見疑問**: {persona.get('communication_style', {}).get('common_questions_type', 'N/A')}
- **表達清晰度**: {persona.get('communication_style', {}).get('clarity_level', 'N/A')}

## 使用場景
- **典型使用時段**: {persona.get('usage_context', {}).get('typical_usage_time', 'N/A')}
- **使用頻率**: {persona.get('usage_context', {}).get('usage_frequency', 'N/A')}
- **使用環境**: {persona.get('usage_context', {}).get('usage_environment', 'N/A')}
- **裝置偏好**: {persona.get('usage_context', {}).get('device_preference', 'N/A')}

---
*Generated by LLM Evaluation Hub - Persona Generator*
*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return md
    
    def _save_personas_excel(self, personas: List[Dict[str, Any]], excel_path: Path):
        """Save personas summary to Excel file."""
        # Flatten personas for Excel
        rows = []
        for persona in personas:
            row = {
                "persona_id": persona.get("persona_id", ""),
                "persona_name": persona.get("persona_name", ""),
                "age": persona.get("basic_info", {}).get("age", ""),
                "gender": persona.get("basic_info", {}).get("gender", ""),
                "occupation": persona.get("basic_info", {}).get("occupation", ""),
                "education": persona.get("basic_info", {}).get("education", ""),
                "income_level": persona.get("basic_info", {}).get("income_level", ""),
                "location": persona.get("basic_info", {}).get("location", ""),
                "marital_status": persona.get("life_status", {}).get("marital_status", ""),
                "family_members": persona.get("life_status", {}).get("family_members", ""),
                "pets": persona.get("life_status", {}).get("pets", ""),
                "pain_points": ", ".join(persona.get("needs_and_goals", {}).get("pain_points", [])),
                "core_needs": ", ".join(persona.get("needs_and_goals", {}).get("core_needs", [])),
                "budget_range": persona.get("needs_and_goals", {}).get("budget_range", ""),
                "decision_style": persona.get("behavior_patterns", {}).get("decision_style", ""),
                "communication_preference": persona.get("behavior_patterns", {}).get("communication_preference", ""),
                "personality_traits": ", ".join(persona.get("psychology", {}).get("personality_traits", [])),
                "values": ", ".join(persona.get("psychology", {}).get("values", []))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved personas summary to Excel: {excel_path}")

