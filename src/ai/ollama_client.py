"""
Ollama Client for HeraAI

Provides integration with the Ollama local AI model API.
"""

import requests
from typing import Optional, Dict, Any
from ..config.settings import AI_CONFIG


class OllamaClient:
    """Client for interacting with Ollama local AI models"""
    
    def __init__(self, model: str = None):
        """
        Initialize the Ollama client
        
        Args:
            model: The model name to use (defaults to configured model)
        """
        self.config = AI_CONFIG["ollama"]
        self.model = model or self.config["default_model"]
        self.base_url = self.config["base_url"]
        self.api_endpoint = self.config["api_endpoint"]
        self.timeout = self.config["timeout"]
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the AI model
        
        Args:
            prompt: The input prompt for the AI
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: The AI's response or error message
        """
        url = f"{self.base_url}{self.api_endpoint}"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.config["stream"]
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "Sorry, I couldn't generate a response.")
            
        except requests.exceptions.ConnectionError:
            return "Sorry, I couldn't connect to the local AI model. Please ensure Ollama is running."
        
        except requests.exceptions.Timeout:
            return "Sorry, the AI model took too long to respond. Please try again."
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API error: {e}")
            return "Sorry, there was an error communicating with the AI model."
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return "Sorry, an unexpected error occurred."
    
    def build_conversation_prompt(self, user_input: str, 
                                context: str = "", 
                                recent_conversation: str = "",
                                has_memory_context: bool = False) -> str:
        """
        Build a comprehensive prompt for conversation
        
        Args:
            user_input: The user's current input
            context: Background context from memory
            recent_conversation: Recent conversation history
            has_memory_context: Whether meaningful memory context exists
            
        Returns:
            str: The formatted prompt for the AI
        """
        prompts = AI_CONFIG["prompts"]
        conversation_config = AI_CONFIG["conversation"]
        
        # Choose appropriate system prompt based on memory availability
        if has_memory_context and context:
            system_prompt = prompts["base_system_prompt"]
            
            # Process context to extract essential information
            background_lines = context.split('\n')
            processed_background = []
            
            for line in background_lines:
                if line.strip():
                    # Keep structure for important memory types
                    if any(mem_type in line for mem_type in ["PERSONAL", "FACT", "PREFERENCE"]):
                        processed_background.append(line)
                    else:
                        # Clean regular conversation lines
                        clean_line = line.replace('[USER]:', '').replace('[AI]:', '').strip()
                        if clean_line and len(clean_line) > conversation_config["min_line_length"]:
                            processed_background.append(f"Previous context: {clean_line}")
            
            if processed_background:
                background_info = f"\nRelevant background about the user:\n" + \
                                "\n".join(processed_background[:conversation_config["max_background_lines"]])
            else:
                background_info = ""
            
            # Build full prompt with context
            full_prompt = f"""{system_prompt}
{background_info}

Recent conversation:
{recent_conversation}

User: {user_input}
AI:"""
        else:
            # No meaningful context available
            system_prompt = prompts["no_memory_prompt"]
            full_prompt = f"""{system_prompt}

Current conversation:
{recent_conversation}

User: {user_input}
AI:"""
        
        return full_prompt
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to get model list as a health check
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
            
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> list:
        """
        Get list of available models
        
        Returns:
            list: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models]
            
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return []
    
    def set_model(self, model: str) -> bool:
        """
        Set the model to use for generation
        
        Args:
            model: The model name
            
        Returns:
            bool: True if model set successfully, False otherwise
        """
        available_models = self.list_models()
        
        if model in available_models:
            self.model = model
            print(f"🔧 AI model changed to: {model}")
            return True
        else:
            print(f"❌ Model '{model}' not available. Available models: {available_models}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {"error": str(e)} 