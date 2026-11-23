import sys
import logging
import os

from src.config import config
from src.prompt import Prompt, BE_SHORT, PYTHON_CODE_ONLY, PATHS_RELATIVE_TO_ROOT
from src.utils import AiClient


def setup_logging() -> None:
    """Configures logging based on the debug flag in config."""
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    # 1. Validate configuration
    if not config.api_key:
        logger.error("API_KEY is not set. Please export it before running.")
        sys.exit(1)

    logger.info(f"Using API URL: {config.api_url}")

    # 2. Create AiClient instance
    model_name = os.environ.get("API_MODEL")
    if not model_name:
        logger.error("API_MODEL is not set. Please export it before running.")
        sys.exit(1)

    ai_client = AiClient(model=model_name)

    # 3. Compose the prompt using the Prompt builder
    base_request = Prompt("Write a Python function that calculates the nth Fibonacci number.")
    
    # Combine prompts: the '+' operator handles priority sorting automatically
    final_prompt = (
        base_request 
        + PATHS_RELATIVE_TO_ROOT 
        + BE_SHORT 
        + PYTHON_CODE_ONLY
    )

    logger.debug(f"Final composed prompt:\n{str(final_prompt)}")

    # 4. Build the API payload (OpenAI-compatible chat completion format)
    logger.info(f"Using MODEL: {model_name}")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": str(final_prompt)
            }
        ],
        "temperature": 0.1,  # Low temperature for code generation
    }

    # 5. Execute the AI call
    try:
        logger.info("Sending request to AI API...")
        response_content = ai_client.call(payload)
        
        print("\n" + "="*50)
        print("AI RESPONSE:")
        print("="*50)
        print(response_content)
        print("="*50 + "\n")
        
    except RuntimeError as e:
        logger.error(f"AI request failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()