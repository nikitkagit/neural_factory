import os
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any

from .config import config

logger = logging.getLogger(__name__)

def call_ai(payload: Dict[str, Any]) -> str:
    """
    Sends a request to the AI API using curl.
    
    :param payload: Dictionary representing the JSON payload of the request.
    :return: String containing the response content.
    """
    logger.debug("Forming payload for API...")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_file = f.name
    
    logger.debug(f"Temporary payload file created: {payload_file}")
    
    try:
        logger.debug("Executing curl request...")
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                config.api_url,
                "-H", f"Authorization: Bearer {config.api_key}",
                "-H", "Content-Type: application/json",
                "-d", f"@{payload_file}",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"curl failed with error: {result.stderr}")
            raise RuntimeError(f"curl failed: {result.stderr}")
        
        logger.debug("Received response from API, parsing JSON...")
        data = json.loads(result.stdout)
        
        # Explicit check for API-level error responses
        if "error" in data:
            logger.error(f"API returned an error: {data['error']}")
            raise RuntimeError(f"API Error: {data['error']}")
        
        # Validate expected response structure
        if "choices" not in data:
            logger.error(f"Unexpected API response format. Full response: {data}")
            raise RuntimeError(f"Unexpected response format (missing 'choices'): {data}")
        
        content = data["choices"][0]["message"].get("content", "")
        
        if not content:
            logger.error(f"Empty response from model. Full response: {data}")
            raise RuntimeError(f"Empty response: {data}")
        
        logger.debug(f"Successfully received {len(content)} characters.")
        return content
    
    finally:
        if os.path.exists(payload_file):
            os.remove(payload_file)
            logger.debug(f"Temporary file {payload_file} deleted.")