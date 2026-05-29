import json
import random
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../app/data/questions.json")

_questions_cache = None

LEVELS_ORDER = ["junior", "mid", "senior"]
TOTAL_QUESTIONS = 3  # 1 por nivel


def load_questions() -> list:
    global _questions_cache
    if _questions_cache is None:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            _questions_cache = json.load(f)
    return _questions_cache


def get_question_by_id(question_id: int) -> dict | None:
    for q in load_questions():
        if q.get("id") == question_id:
            return q
    return None


def get_interview_questions(category: str | None = None) -> list[int]:
    """Selección balanceada: 1 junior + 1 mid + 1 senior."""
    qs = load_questions()
    result = []
    used: list[int] = []
    for level in LEVELS_ORDER:
        pool = [q for q in qs if q["difficulty"] == level and q["id"] not in used]
        if category and category != "all":
            cat_pool = [q for q in pool if q["category"] == category]
            if len(cat_pool) >= 1:
                pool = cat_pool
        selected = random.sample(pool, min(1, len(pool)))
        ids = [q["id"] for q in selected]
        result.extend(ids)
        used.extend(ids)
    return result


def get_available_categories() -> list[str]:
    qs = load_questions()
    return sorted(set(q["category"] for q in qs))


def get_stats() -> dict:
    qs = load_questions()
    stats = {"total": len(qs), "by_level": {}, "by_category": {}}
    for q in qs:
        lvl, cat = q["difficulty"], q["category"]
        stats["by_level"][lvl] = stats["by_level"].get(lvl, 0) + 1
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
    return stats
