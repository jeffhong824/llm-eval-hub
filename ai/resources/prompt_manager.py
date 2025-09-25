"""Prompt manager for loading and managing system prompts."""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """Manager for loading and accessing system prompts."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files. Defaults to ai/resources/prompts
        """
        if prompts_dir is None:
            # Get the directory of this file and navigate to prompts
            current_dir = Path(__file__).parent
            prompts_dir = current_dir / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._prompts_cache: Dict[str, str] = {}
        
        logger.info(f"PromptManager initialized with prompts directory: {self.prompts_dir}")
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from file.
        
        Args:
            prompt_name: Name of the prompt file (without extension)
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if prompt_name in self._prompts_cache:
            return self._prompts_cache[prompt_name]
        
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            self._prompts_cache[prompt_name] = content
            logger.info(f"Loaded prompt: {prompt_name}")
            return content
            
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {str(e)}")
            raise
    
    def get_scenario_to_documents_prompt(self) -> str:
        """Get the scenario to documents system prompt."""
        return self.load_prompt("scenario_to_documents")
    
    def get_qa_generation_prompt(self) -> str:
        """Get the QA generation system prompt."""
        return self.load_prompt("qa_generation")
    
    def get_agent_task_generation_prompt(self) -> str:
        """Get the agent task generation system prompt."""
        return self.load_prompt("agent_task_generation")
    
    def list_available_prompts(self) -> list:
        """List all available prompt files."""
        if not self.prompts_dir.exists():
            return []
        
        prompt_files = []
        for file_path in self.prompts_dir.glob("*.txt"):
            prompt_files.append(file_path.stem)
        
        return sorted(prompt_files)
    
    def reload_prompts(self):
        """Clear cache and reload all prompts."""
        self._prompts_cache.clear()
        logger.info("Prompt cache cleared")


# Global instance
prompt_manager = PromptManager()
