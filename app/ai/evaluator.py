"""
Motor de evaluación principal.
Combina:
  1. Embeddings semánticos (similitud coseno)
  2. LLM (Ollama) para feedback cualitativo
  3. Métricas compuestas ponderadas
  4. RAG para contexto adicional
"""
from __future__ import annotations
import json
import re
import time

from app.ai.ollama_client import chat_ollama
from app.ai.embeddings import compute_similarity
from app.ai.metrics import (
    calculate_keyword_coverage,
    calculate_quality_score,
    classify_score,
    classify_quality,
)
from app.ai.prompts import EVALUATOR_PROMPT, FOLLOW_UP_PROMPT


def _extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON válido de un texto."""
    # Intento directo
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Busca bloque ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Busca primer { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No se pudo extraer JSON de la respuesta del LLM:\n{text[:500]}")


def evaluate_answer(
    question: str,
    ideal_answer: str,
    user_answer: str,
    keywords: list[str] | None = None,
    question_id: int | None = None,
    code_snippet: str | None = None,
) -> dict:
    """
    Evalúa la respuesta del usuario.

    Retorna un dict con todos los scores, métricas y feedback.
    """
    if keywords is None:
        keywords = []

    start_time = time.time()

    # 1. Similitud semántica con embeddings
    semantic_sim = compute_similarity(user_answer, ideal_answer)
    semantic_label = classify_score(semantic_sim)

    # 2. Cobertura de keywords (máx 5 para no penalizar listas largas)
    score_kws = keywords[:5]
    kw_coverage = calculate_keyword_coverage(user_answer, score_kws)
    keywords_found = [kw for kw in score_kws if kw.lower() in user_answer.lower()]
    keywords_missing = [kw for kw in score_kws if kw.lower() not in user_answer.lower()]

    # 3. Evaluación LLM
    question_with_context = question
    if code_snippet:
        question_with_context = f"{question}\n\nCódigo de referencia:\n```\n{code_snippet}\n```"

    prompt = EVALUATOR_PROMPT.format(
        question=question_with_context,
        ideal_answer=ideal_answer,
        keywords=", ".join(keywords),
        user_answer=user_answer,
    )

    llm_raw = chat_ollama(
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        llm_data = _extract_json(llm_raw)
    except ValueError:
        # Fallback si el LLM no devuelve JSON limpio
        llm_data = {
            "score": round(semantic_sim * 10),
            "level_detected": "mid",
            "semantic_similarity": semantic_sim,
            "keyword_coverage": kw_coverage,
            "keywords_found": keywords_found,
            "keywords_missing": keywords_missing,
            "strengths": "No se pudo parsear la respuesta del modelo.",
            "improvements": "Inténtalo de nuevo.",
            "feedback": llm_raw[:500],
        }

    # Usar los valores del LLM donde tenga sentido, o los calculados localmente
    llm_score = int(llm_data.get("score", round(semantic_sim * 10)))
    llm_score = max(1, min(10, llm_score))  # clamp 1-10

    # Sobreescribir con métricas calculadas localmente (más confiables)
    llm_data["semantic_similarity"] = semantic_sim
    llm_data["semantic_label"] = semantic_label
    llm_data["keyword_coverage"] = kw_coverage
    llm_data["keywords_found"] = keywords_found
    llm_data["keywords_missing"] = keywords_missing
    llm_data["score"] = llm_score

    # 4. Score compuesto
    quality = calculate_quality_score(kw_coverage, semantic_sim, llm_score, user_answer)
    llm_data.update(quality)
    llm_data["quality_label"] = classify_quality(quality["quality_percent"])

    # 5. Tiempo de respuesta
    elapsed = round(time.time() - start_time, 2)
    llm_data["response_time_seconds"] = elapsed

    return llm_data


def generate_follow_up(question: str, user_answer: str, score: int) -> str:
    """Genera una pregunta de seguimiento basada en la respuesta del candidato."""
    prompt = FOLLOW_UP_PROMPT.format(
        question=question,
        user_answer=user_answer,
        score=score,
    )
    return chat_ollama(messages=[{"role": "user", "content": prompt}])
