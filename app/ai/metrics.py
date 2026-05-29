"""
Módulo de métricas de evaluación.
Calcula las 4 métricas ponderadas del sistema:
  - Relevancia (w=0.25): cobertura de keywords
  - Coherencia (w=0.20): longitud y estructura como proxy
  - Cobertura (w=0.30): similitud semántica
  - Exactitud (w=0.25): score del LLM normalizado
"""
from __future__ import annotations


def calculate_keyword_coverage(user_answer: str, keywords: list[str]) -> float:
    """Fracción de keywords cubiertas, normalizada: 60% de cobertura = score 1.0."""
    if not keywords:
        return 0.0
    ua_lower = user_answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in ua_lower)
    raw = found / len(keywords)
    return round(min(raw / 0.6, 1.0), 4)


def calculate_coherence(user_answer: str) -> float:
    """
    Proxy de coherencia basado en longitud y presencia de conectores lógicos.
    Escala 0-1.
    """
    if not user_answer or len(user_answer.strip()) < 10:
        return 0.0

    words = user_answer.split()
    word_count = len(words)

    # Longitud óptima: entre 50 y 300 palabras → score 1.0
    if word_count < 10:
        length_score = 0.2
    elif word_count < 30:
        length_score = 0.5
    elif word_count <= 300:
        length_score = 1.0
    else:
        # Penaliza respuestas muy largas (posible divagación)
        length_score = max(0.6, 1.0 - (word_count - 300) / 500)

    connectors = [
        "porque", "ya que", "por lo tanto", "sin embargo", "además",
        "por ejemplo", "es decir", "en cambio", "mientras que", "también",
        "aunque", "pero", "cuando", "entonces", "finalmente", "primero"
    ]
    ua_lower = user_answer.lower()
    connector_score = min(1.0, sum(1 for c in connectors if c in ua_lower) / 3)

    return round((length_score * 0.6 + connector_score * 0.4), 4)


def calculate_quality_score(
    keyword_coverage: float,
    semantic_similarity: float,
    llm_score: int,
    user_answer: str,
) -> dict:
    """
    Fórmula compuesta:
    Calidad = 0.25*Relevancia + 0.20*Coherencia + 0.30*Cobertura + 0.25*Exactitud
    """
    relevance = keyword_coverage                        # w=0.25
    coherence = calculate_coherence(user_answer)        # w=0.20
    coverage = semantic_similarity                      # w=0.30
    accuracy = round(llm_score / 10, 4)                 # w=0.25 (normalizado)

    quality = (
        0.25 * relevance
        + 0.20 * coherence
        + 0.30 * coverage
        + 0.25 * accuracy
    )

    return {
        "relevance": relevance,
        "coherence": coherence,
        "coverage": coverage,
        "accuracy": accuracy,
        "quality_score": round(quality, 4),
        "quality_percent": round(quality * 100, 1),
    }


def classify_score(score: float) -> str:
    """Clasifica el score de similitud semántica (escala 0-1)."""
    if score >= 0.85:
        return "Excelente"
    elif score >= 0.70:
        return "Bueno"
    elif score >= 0.50:
        return "Aceptable"
    else:
        return "Insuficiente"


def classify_quality(quality_percent: float) -> str:
    """Clasifica el score de calidad compuesta (escala 0-100)."""
    if quality_percent >= 80:
        return "Excelente"
    elif quality_percent >= 65:
        return "Bueno"
    elif quality_percent >= 45:
        return "Aceptable"
    else:
        return "Insuficiente"
