"""
Módulo de embeddings semánticos.
Usa sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) si está disponible.
Si no, usa un fallback basado en overlap de palabras (TF-IDF simplificado).
"""
from __future__ import annotations
import math

_model = None
_use_transformers = False


def _load_model():
    global _model, _use_transformers
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _use_transformers = True
    except Exception:
        _use_transformers = False


def _cosine_similarity_numpy(vec_a, vec_b) -> float:
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _fallback_similarity(text_a: str, text_b: str) -> float:
    """
    Similitud por overlap de tokens (Jaccard + peso TF).
    Es un fallback razonable cuando no hay GPU/sentence-transformers.
    """
    def tokenize(text: str) -> dict[str, int]:
        stopwords = {
            "el", "la", "los", "las", "un", "una", "de", "del", "en",
            "es", "son", "se", "que", "y", "o", "a", "con", "por",
            "para", "como", "más", "pero", "si", "no", "al", "su",
            "sus", "lo", "le", "les", "this", "the", "is", "are",
            "of", "in", "to", "and", "or", "a", "an"
        }
        tokens = text.lower().split()
        freq: dict[str, int] = {}
        for t in tokens:
            t = t.strip(".,;:()[]\"'")
            if len(t) > 2 and t not in stopwords:
                freq[t] = freq.get(t, 0) + 1
        return freq

    freq_a = tokenize(text_a)
    freq_b = tokenize(text_b)
    if not freq_a or not freq_b:
        return 0.0

    set_a = set(freq_a.keys())
    set_b = set(freq_b.keys())
    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    # Jaccard ponderado por frecuencia
    intersection_score = sum(min(freq_a[t], freq_b[t]) for t in intersection)
    union_score = sum(max(freq_a.get(t, 0), freq_b.get(t, 0)) for t in union)

    return round(intersection_score / union_score, 4) if union_score > 0 else 0.0


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Calcula similitud semántica entre dos textos.
    Retorna valor entre 0.0 y 1.0.
    """
    _load_model()

    if not text_a.strip() or not text_b.strip():
        return 0.0

    if _use_transformers and _model is not None:
        try:
            embeddings = _model.encode([text_a, text_b])
            sim = _cosine_similarity_numpy(embeddings[0], embeddings[1])
            return round(float(sim), 4)
        except Exception:
            pass

    return _fallback_similarity(text_a, text_b)
