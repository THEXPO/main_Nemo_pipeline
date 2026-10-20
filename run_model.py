"""
run_model.py
Connects to a locally running model served via LM Studio's OpenAI-compatible REST API.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import requests

# Set this to match the exact model identifier shown in LM Studio's "Local Server" tab
# e.g., "mistralai/mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", or "local-model"
MODEL_NAME = "local-model"

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
CONNECT_TIMEOUT = 0.5  # seconds to establish TCP connection (fails fast if server is offline)
READ_TIMEOUT = 60.0    # seconds to wait for model generation response


def check_lm_studio_server(base_url: str = BASE_URL) -> Tuple[bool, str, List[str]]:
    """
    Pings LM Studio's /v1/models endpoint to verify the local server is running.
    Returns: (is_reachable: bool, status_message: str, loaded_models: List[str])
    """
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        resp = requests.get(models_url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id") for m in data.get("data", []) if "id" in m]
            return True, "LM Studio server is running and reachable.", models
        else:
            return False, f"Server responded with HTTP {resp.status_code}: {resp.text[:200]}", []
    except requests.exceptions.ConnectionError:
        return (
            False,
            f"Could not connect to LM Studio at {models_url}. "
            "Please ensure LM Studio is open, a model is loaded, and the local server is started (port 1234).",
            [],
        )
    except requests.exceptions.Timeout:
        return False, f"Connection timed out while pinging {models_url}.", []
    except Exception as e:
        return False, f"Unexpected error while checking LM Studio: {e}", []


def generate(prompt: str, model_name: str = MODEL_NAME, max_tokens: int = 256) -> Dict[str, Any]:
    """
    Sends prompt as a single user message to LM Studio's /v1/chat/completions endpoint.

    Returns:
        {
            "output_text": str,
            "latency_seconds": float,
            "tokens_out": int,
            "error": Optional[str]
        }
    """
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        latency = round(time.perf_counter() - start_time, 3)

        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text[:300]}"
            print(f"    [API Error] {error_msg}")
            return {
                "output_text": "",
                "latency_seconds": latency,
                "tokens_out": 0,
                "error": error_msg,
            }

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            error_msg = f"No choices returned in API response: {data}"
            print(f"    [API Error] {error_msg}")
            return {
                "output_text": "",
                "latency_seconds": latency,
                "tokens_out": 0,
                "error": error_msg,
            }

        output_text = choices[0].get("message", {}).get("content", "").strip()

        # Token count from usage field, fallback to whitespace token count
        usage = data.get("usage", {})
        if "completion_tokens" in usage and usage["completion_tokens"] is not None:
            tokens_out = int(usage["completion_tokens"])
        else:
            tokens_out = len(output_text.split())

        return {
            "output_text": output_text,
            "latency_seconds": latency,
            "tokens_out": tokens_out,
            "error": None,
        }

    except requests.exceptions.ConnectionError as ce:
        latency = round(time.perf_counter() - start_time, 3)
        err = f"Connection failed to LM Studio at {url}: {ce}"
        print(f"    [Connection Error] {err}")
        return {
            "output_text": "",
            "latency_seconds": latency,
            "tokens_out": 0,
            "error": err,
        }
    except requests.exceptions.Timeout as te:
        latency = round(time.perf_counter() - start_time, 3)
        err = f"Request timed out after {READ_TIMEOUT}s: {te}"
        print(f"    [Timeout Error] {err}")
        return {
            "output_text": "",
            "latency_seconds": latency,
            "tokens_out": 0,
            "error": err,
        }
    except Exception as e:
        latency = round(time.perf_counter() - start_time, 3)
        err = f"Unexpected generation error: {e}"
        print(f"    [Error] {err}")
        return {
            "output_text": "",
            "latency_seconds": latency,
            "tokens_out": 0,
            "error": err,
        }


if __name__ == "__main__":
    print(f"Targeting MODEL_NAME: '{MODEL_NAME}' at {BASE_URL}")
    reachable, msg, models = check_lm_studio_server()
    print(f"Status: {msg}")
    if reachable:
        print(f"Available models in LM Studio: {models}")
        print("Testing sample generation call...")
        res = generate("Respond with exactly: 'LM Studio connection successful!'")
        print("Result:", res)
    else:
        print("\nTo run tests with LM Studio:")
        print("  1. Launch LM Studio")
        print("  2. Load a model")
        print("  3. Go to Local Server tab and click 'Start Server'")
