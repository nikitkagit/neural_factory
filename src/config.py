# config.py
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self) -> None:
        self._api_key = os.environ.get("API_KEY", "")
        self._api_url = os.environ.get("API_URL", "https://routerai.ru/api/v1/chat/completions")
        self._debug = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

        # Validate API key presence and format
        if not self._api_key or not self._api_key.strip():
            logger.warning("API_KEY is missing or empty in environment variables.")
        elif len(self._api_key) < 10 or not self._api_key.startswith("sk-"):
            logger.warning("API_KEY appears to be invalid or malformed.")

        # Validate API URL presence
        if not self._api_url or not self._api_url.strip():
            logger.warning("API_URL is missing or empty in environment variables.")

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def debug(self) -> bool:
        return self._debug

config = Config()