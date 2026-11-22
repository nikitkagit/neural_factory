import re
import os
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any, List

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


def recalculate_diff_headers(diff_text: str) -> str:
    """
    Recalculates hunk headers (@@ -X,Y +X,Y @@) in a unified diff string.

    Useful when an LLM generates a diff with incorrect line counts.
    Preserves original start line numbers and recomputes only the counts.

    :param diff_text: The raw unified diff as a single string.
    :return: The diff with corrected hunk headers.
    """
    lines = diff_text.splitlines(keepends=True)
    result: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect hunk header — match only the @@ ... @@ part
        match = re.match(r"^(@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@)(.*)", line)
        if not match:
            result.append(line)
            i += 1
            continue

        header_part = match.group(1)
        old_start = int(match.group(2))
        new_start = int(match.group(4))
        suffix = match.group(6) or ""

        # Collect hunk body lines (until next @@ or end of file)
        hunk_body: List[str] = []
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            if next_line.startswith("@@ ") or next_line.startswith("diff "):
                break
            hunk_body.append(next_line)
            j += 1

        # Count lines in old and new versions
        old_count = 0
        new_count = 0
        for body_line in hunk_body:
            stripped = body_line.rstrip("\n").rstrip("\r")
            if stripped.startswith("\\"):
                continue
            if stripped.startswith("-"):
                old_count += 1
            elif stripped.startswith("+"):
                new_count += 1
            elif stripped.startswith(" "):
                old_count += 1
                new_count += 1
            # Empty lines inside a hunk — skip, they are not valid diff lines
            # and should not affect counts

        # Rebuild header preserving the suffix (e.g., class name)
        new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}\n"
        result.append(new_header)
        result.extend(hunk_body)

        i = j

    return "".join(result)