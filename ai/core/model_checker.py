"""Model availability checker for different LLM providers."""

import logging
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information."""
    name: str
    provider: str
    available: bool
    error_message: Optional[str] = None


class ModelChecker:
    """Check availability of different LLM models."""
    
    def __init__(self):
        """Initialize model checker."""
        self.openai_models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-05-13",
            "o1-preview",
            "o1-mini"
        ]
        
        self.gemini_models = [
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b"
        ]
        
        self.ollama_models = [
            "llama2",
            "llama2:7b",
            "llama2:13b",
            "llama2:70b",
            "qwen2.5:0.5b",
            "codellama",
            "mistral",
            "neural-chat",
            "starling-lm",
            "orca-mini"
        ]
    
    async def check_openai_models(self) -> List[ModelInfo]:
        """Check OpenAI model availability."""
        models = []
        
        if not settings.openai_api_key:
            for model_name in self.openai_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="openai",
                    available=False,
                    error_message="OpenAI API key not configured"
                ))
            return models
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Get available models from OpenAI API
            try:
                available_models = client.models.list()
                available_model_names = [model.id for model in available_models.data]
                
                for model_name in self.openai_models:
                    if model_name in available_model_names:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="openai",
                            available=True
                        ))
                    else:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="openai",
                            available=False,
                            error_message=f"Model not available in OpenAI API"
                        ))
                        
            except Exception as e:
                logger.error(f"Error fetching OpenAI models: {str(e)}")
                # Fallback to testing individual models
                for model_name in self.openai_models:
                    try:
                        llm = ChatOpenAI(
                            openai_api_key=settings.openai_api_key,
                            model_name=model_name,
                            temperature=0.1,
                            max_tokens=10
                        )
                        
                        # Test with a simple request
                        response = await llm.ainvoke("test")
                        
                        models.append(ModelInfo(
                            name=model_name,
                            provider="openai",
                            available=True
                        ))
                        
                    except Exception as e:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="openai",
                            available=False,
                            error_message=str(e)
                        ))
                    
        except Exception as e:
            logger.error(f"Error checking OpenAI models: {str(e)}")
            for model_name in self.openai_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="openai",
                    available=False,
                    error_message=f"OpenAI client error: {str(e)}"
                ))
        
        return models
    
    async def check_gemini_models(self) -> List[ModelInfo]:
        """Check Gemini model availability."""
        models = []
        
        if not settings.gemini_api_key:
            for model_name in self.gemini_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="gemini",
                    available=False,
                    error_message="Gemini API key not configured"
                ))
            return models
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            
            # Get available models from Gemini API
            try:
                available_models = list(genai.list_models())
                available_model_names = [model.name for model in available_models]
                
                for model_name in self.gemini_models:
                    # Check if model is available
                    full_model_name = f"models/{model_name}"
                    if full_model_name in available_model_names:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="gemini",
                            available=True
                        ))
                    else:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="gemini",
                            available=False,
                            error_message=f"Model not available in Gemini API"
                        ))
                        
            except Exception as e:
                logger.error(f"Error fetching Gemini models: {str(e)}")
                # Fallback to testing individual models
                for model_name in self.gemini_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content("test")
                        
                        models.append(ModelInfo(
                            name=model_name,
                            provider="gemini",
                            available=True
                        ))
                        
                    except Exception as e:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="gemini",
                            available=False,
                            error_message=str(e)
                        ))
                    
        except Exception as e:
            logger.error(f"Error checking Gemini models: {str(e)}")
            for model_name in self.gemini_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="gemini",
                    available=False,
                    error_message=f"Gemini client error: {str(e)}"
                ))
        
        return models
    
    async def check_ollama_models(self) -> List[ModelInfo]:
        """Check Ollama model availability."""
        models = []
        
        if not settings.ollama_base_url:
            for model_name in self.ollama_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="ollama",
                    available=False,
                    error_message="Ollama base URL not configured"
                ))
            return models
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Check if Ollama is running and get available models
                try:
                    # Try both localhost and container name
                    urls_to_try = [
                        f"{settings.ollama_base_url}/api/tags",
                        "http://ollama:11434/api/tags"
                    ]
                    
                    response = None
                    for url in urls_to_try:
                        try:
                            response = await client.get(url, timeout=5.0)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to connect to {url}: {str(e)}")
                            continue
                    
                    if response is None:
                        raise Exception("Could not connect to Ollama server")
                    if response.status_code != 200:
                        raise Exception(f"Ollama server not responding: {response.status_code}")
                    
                    available_models = response.json().get("models", [])
                    available_model_names = [model["name"] for model in available_models]
                    
                    # Check each predefined model
                    for model_name in self.ollama_models:
                        # Check exact match first
                        if model_name in available_model_names:
                            models.append(ModelInfo(
                                name=model_name,
                                provider="ollama",
                                available=True
                            ))
                        else:
                            # Check for partial matches (e.g., "llama2" matches "llama2:7b")
                            found_match = False
                            for available_name in available_model_names:
                                if available_name.startswith(model_name + ":") or available_name == model_name:
                                    models.append(ModelInfo(
                                        name=model_name,
                                        provider="ollama",
                                        available=True,
                                        error_message=f"Available as: {available_name}"
                                    ))
                                    found_match = True
                                    break
                            
                            if not found_match:
                                models.append(ModelInfo(
                                    name=model_name,
                                    provider="ollama",
                                    available=False,
                                    error_message=f"Model not found in Ollama. Available models: {', '.join(available_model_names[:5])}"
                                ))
                            
                except Exception as e:
                    for model_name in self.ollama_models:
                        models.append(ModelInfo(
                            name=model_name,
                            provider="ollama",
                            available=False,
                            error_message=f"Ollama connection error: {str(e)}"
                        ))
                        
        except Exception as e:
            logger.error(f"Error checking Ollama models: {str(e)}")
            for model_name in self.ollama_models:
                models.append(ModelInfo(
                    name=model_name,
                    provider="ollama",
                    available=False,
                    error_message=f"Ollama client error: {str(e)}"
                ))
        
        return models
    
    async def check_all_models(self) -> Dict[str, List[ModelInfo]]:
        """Check all model availability."""
        logger.info("Checking model availability...")
        
        # Run all checks concurrently
        openai_models, gemini_models, ollama_models = await asyncio.gather(
            self.check_openai_models(),
            self.check_gemini_models(),
            self.check_ollama_models(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(openai_models, Exception):
            logger.error(f"OpenAI check failed: {openai_models}")
            openai_models = []
        
        if isinstance(gemini_models, Exception):
            logger.error(f"Gemini check failed: {gemini_models}")
            gemini_models = []
        
        if isinstance(ollama_models, Exception):
            logger.error(f"Ollama check failed: {ollama_models}")
            ollama_models = []
        
        return {
            "openai": openai_models,
            "gemini": gemini_models,
            "ollama": ollama_models
        }
    
    def get_available_models_summary(self, models_info: Dict[str, List[ModelInfo]]) -> Dict[str, List[str]]:
        """Get summary of available models."""
        summary = {}
        
        for provider, models in models_info.items():
            available_models = [model.name for model in models if model.available]
            summary[provider] = available_models
        
        return summary


# Global instance
model_checker = ModelChecker()
