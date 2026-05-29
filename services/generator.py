import json
import re
from app.ai.ollama_client import chat_ollama
from app.ai.prompts import QUESTION_GENERATOR_PROMPT


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("El LLM no devolvió JSON válido")


def generate_question(category: str, difficulty: str) -> dict:
    """Genera una pregunta nueva con Ollama (no del dataset)."""
    prompt = QUESTION_GENERATOR_PROMPT.format(
        category=category,
        difficulty=difficulty
    )
    raw = chat_ollama(messages=[{"role": "user", "content": prompt}])
    try:
        return _extract_json(raw)
    except ValueError:
        return {
            "question": f"[Error al generar] Explica un concepto clave de {category} nivel {difficulty}.",
            "ideal_answer": "No se pudo generar una respuesta de referencia.",
            "keywords": [],
            "difficulty": difficulty,
            "category": category,
            "generated": True,
            "error": True,
        }
