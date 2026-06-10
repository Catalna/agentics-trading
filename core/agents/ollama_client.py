import json
import logging
import requests
import time
from typing import Dict, Any, Optional

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = config.OLLAMA_URL):
        self.base_url = base_url
        self.api_generate = f"{self.base_url}/api/generate"

    def generate(self, model: str, prompt: str, system: str = "", timeout: int = config.LLM_TIMEOUT_S, format: str = "json") -> Optional[Dict[str, Any]]:
        """
        Calls the Ollama API synchronously.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": format
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.api_generate, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            response_text = data.get("response", "{}")
            
            # Since format="json", the output should be valid JSON
            try:
                result_json = json.loads(response_text)
                elapsed = time.time() - start_time
                logger.debug(f"Ollama {model} call successful in {elapsed:.2f}s")
                return result_json
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {model}. Output: {response_text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Ollama call to {model} timed out after {timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error for {model}: {e}")
            return None
