import os
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def chat_ollama(messages: list, model: str = DEFAULT_MODEL) -> str:
    """Envía mensajes al endpoint /api/chat de Ollama y devuelve el texto."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"No se pudo conectar a Ollama en {OLLAMA_BASE_URL}. "
            "Ejecuta 'ollama serve' y asegúrate de que el modelo esté descargado: "
            f"'ollama pull {model}'"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama tardó demasiado en responder (timeout 120s).")
    except Exception as e:
        raise RuntimeError(f"Error al comunicarse con Ollama: {e}")


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    return chat_ollama(messages=[{"role": "user", "content": prompt}], model=model)


def ollama_health_check() -> dict:
    """Verifica si Ollama está corriendo y el modelo está disponible."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            model_ok = any(DEFAULT_MODEL.split(":")[0] in m for m in models)
            return {
                "ok": model_ok,
                "models": models,
                "active_model": DEFAULT_MODEL,
                "warning": None if model_ok else f"Modelo '{DEFAULT_MODEL}' no encontrado. Ejecuta: ollama pull {DEFAULT_MODEL}",
            }
    except Exception:
        pass
    return {
        "ok": False,
        "models": [],
        "active_model": DEFAULT_MODEL,
        "warning": "Ollama no está corriendo. Ejecuta: ollama serve",
    }
