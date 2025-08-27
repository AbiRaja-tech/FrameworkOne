# ollama_client.py
"""
Ollama client for DeepSeek model integration.
This provides a proper API client for calling DeepSeek through Ollama.
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ollama_client")

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-coder:33b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (default: localhost:11434)
            model: Model name to use (default: deepseek-coder:33b)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Ollama API request failed: {e}")
    
    def generate(self, 
                 prompt: str, 
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048) -> str:
        """
        Generate text using the specified model.
        
        Args:
            prompt: User prompt
            system: System message (optional)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Build messages array for chat API
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = self._make_request("/api/chat", data)
            return response.get("message", {}).get("content", "")
        except Exception as e:
            log.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    def test_connection(self) -> bool:
        """Test if Ollama server is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return []
        except:
            return []

def create_deepseek_client() -> OllamaClient:
    """Create a DeepSeek client with appropriate defaults."""
    # Try to use available models
    client = OllamaClient()
    available_models = client.list_models()
    
    # Prefer DeepSeek, fallback to Mistral if available
    if "deepseek-coder:33b" in available_models:
        return OllamaClient(model="deepseek-coder:33b")
    elif "mistral:latest" in available_models:
        log.info("Using Mistral as fallback for DeepSeek")
        return OllamaClient(model="mistral:latest")
    else:
        log.warning("No preferred models found, using default")
        return OllamaClient()

if __name__ == "__main__":
    # Test the client
    client = create_deepseek_client()
    
    if client.test_connection():
        print("✅ Ollama connection successful!")
        models = client.list_models()
        print(f"Available models: {models}")
        print(f"Using model: {client.model}")
        
        # Test generation
        try:
            response = client.generate(
                prompt="Hello, can you help me with travel planning?",
                system="You are a helpful travel assistant.",
                temperature=0.7,
                max_tokens=100
            )
            print(f"Test response: {response}")
        except Exception as e:
            print(f"Test generation failed: {e}")
    else:
        print("❌ Ollama connection failed. Make sure Ollama is running on localhost:11434")
