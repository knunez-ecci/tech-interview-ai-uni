"""
Benchmark comparativo de modelos LLM — Tech Interview AI (Universidad)
======================================================================
Métricas evaluadas por modelo:
  1. Exactitud      — score del modelo vs score esperado (ground truth)
  2. Alucinaciones  — términos técnicos inventados no presentes en el input
  3. Consistencia   — varianza del score entre ejecuciones del mismo caso
  4. Velocidad      — tiempo promedio de respuesta (segundos)
  5. JSON válido    — tasa de respuestas bien estructuradas

Uso:
    python benchmark.py
    python benchmark.py --models qwen2.5:3b llama3.2:3b gemma3:4b mistral:7b
    python benchmark.py --runs 3 --export
"""

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests

# ─── Configuración ────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemma3:4b",
    "qwen2.5:7b",
]

OLLAMA_URL  = "http://localhost:11434"
TIMEOUT     = 180
RUNS        = 1   # repeticiones por caso (más = mayor confianza estadística)

# ─── Ground truth: casos de prueba con score esperado conocido ────────────────
# expected_score: [min, max] para cada calidad de respuesta
# hallucination_check_terms: palabras técnicas que NO deben aparecer en feedback
#   si el modelo no las vio en el input (miden confabulación)

TEST_CASES = [
    {
        "id": "TC-01",
        "topic": "REST API",
        "question": "¿Qué es una API REST y cuáles son sus principios fundamentales?",
        "ideal_answer": (
            "REST (Representational State Transfer) es un estilo arquitectónico para APIs web. "
            "Sus principios son: sin estado (stateless), interfaz uniforme, arquitectura cliente-servidor, "
            "sistema en capas, caché y código bajo demanda. Usa verbos HTTP (GET, POST, PUT, DELETE) "
            "y recursos identificados por URLs."
        ),
        "keywords": ["stateless", "HTTP", "recursos", "GET", "POST", "cliente-servidor"],
        "forbidden_terms": ["GraphQL", "SOAP", "gRPC", "WebSocket", "OAuth"],
        "answers": [
            {
                "label": "completa",
                "expected": [7, 10],
                "text": (
                    "REST es un estilo arquitectónico para diseñar APIs web basado en HTTP. "
                    "Sus principios fundamentales incluyen ser stateless (sin estado entre peticiones), "
                    "usar verbos HTTP como GET, POST, PUT y DELETE, una interfaz uniforme, "
                    "arquitectura cliente-servidor, soporte de caché y sistema en capas. "
                    "Los recursos se identifican con URLs únicas."
                ),
            },
            {
                "label": "parcial",
                "expected": [3, 6],
                "text": (
                    "Una API REST usa HTTP para comunicar sistemas. "
                    "Usa GET para obtener datos y POST para enviarlos. "
                    "Es muy usada en aplicaciones web y móviles."
                ),
            },
            {
                "label": "incorrecta",
                "expected": [1, 3],
                "text": "Es una API que conecta aplicaciones en internet.",
            },
        ],
    },
    {
        "id": "TC-03",
        "topic": "Índices SQL",
        "question": "¿Qué es un índice en SQL y cuándo deberías y no deberías usarlo?",
        "ideal_answer": (
            "Un índice en SQL es una estructura de datos (B-tree por defecto) que acelera las consultas "
            "permitiendo búsquedas sin escanear toda la tabla. Se usa en columnas con alta cardinalidad "
            "en WHERE, JOIN u ORDER BY frecuentes. No conviene en tablas pequeñas, columnas con baja "
            "cardinalidad o tablas con muchas escrituras, ya que los índices ralentizan INSERT/UPDATE/DELETE "
            "y consumen almacenamiento adicional."
        ),
        "keywords": ["B-tree", "cardinalidad", "WHERE", "JOIN", "INSERT", "almacenamiento"],
        "forbidden_terms": ["NoSQL", "MongoDB", "Redis", "Elasticsearch", "partición", "sharding"],
        "answers": [
            {
                "label": "completa",
                "expected": [7, 10],
                "text": (
                    "Un índice SQL es una estructura B-tree que permite encontrar registros "
                    "sin escanear la tabla completa. Úsalo en columnas con alta cardinalidad "
                    "que aparecen frecuentemente en WHERE, JOIN o ORDER BY. "
                    "No lo uses en tablas pequeñas, columnas con poca cardinalidad o tablas "
                    "con muchas operaciones de escritura, ya que los índices ralentizan "
                    "INSERT, UPDATE y DELETE, además de consumir almacenamiento adicional."
                ),
            },
            {
                "label": "parcial",
                "expected": [3, 6],
                "text": (
                    "Un índice en SQL hace más rápidas las consultas. "
                    "Se crea sobre columnas que se buscan frecuentemente. "
                    "El problema es que ocupan espacio y pueden hacer más lentos los inserts."
                ),
            },
            {
                "label": "incorrecta",
                "expected": [1, 3],
                "text": "Es algo que se pone en las tablas para buscar más rápido.",
            },
        ],
    },
    {
        "id": "TC-05",
        "topic": "Big O",
        "question": "Explica la notación Big O y da ejemplos de O(1), O(n) y O(n²).",
        "ideal_answer": (
            "Big O es una notación matemática que describe la complejidad temporal o espacial de un "
            "algoritmo en el peor caso, en función del tamaño del input n. "
            "O(1) es constante: acceder a un elemento de un array por índice. "
            "O(n) es lineal: recorrer una lista completa. "
            "O(n²) es cuadrática: dos bucles anidados iterando sobre el mismo conjunto, "
            "como el algoritmo bubble sort."
        ),
        "keywords": ["complejidad", "peor caso", "O(1)", "O(n)", "O(n²)", "algoritmo", "input"],
        "forbidden_terms": ["O(log n)", "quicksort", "mergesort", "árbol", "hash", "recursión"],
        "answers": [
            {
                "label": "completa",
                "expected": [7, 10],
                "text": (
                    "Big O describe la complejidad de un algoritmo en función del tamaño del "
                    "input n, representando el peor caso. O(1) es constante: acceder a un "
                    "array por índice no depende del tamaño. O(n) es lineal: recorrer una "
                    "lista entera. O(n²) es cuadrática: dos bucles anidados sobre el mismo "
                    "conjunto de datos, como bubble sort."
                ),
            },
            {
                "label": "parcial",
                "expected": [3, 6],
                "text": (
                    "Big O mide qué tan rápido es un algoritmo. O(1) es muy rápido porque "
                    "siempre tarda igual. O(n) depende del tamaño de los datos. "
                    "O(n²) es lento porque tiene bucles dentro de bucles."
                ),
            },
            {
                "label": "incorrecta",
                "expected": [1, 3],
                "text": "Big O es una forma de medir la velocidad de los programas.",
            },
        ],
    },
]

PROMPT = """Eres un entrevistador técnico senior. Evalúa la respuesta del candidato.

Pregunta: {question}
Respuesta ideal: {ideal_answer}
Palabras clave esperadas: {keywords}
Respuesta del candidato: {user_answer}

Responde ÚNICAMENTE con este JSON válido, sin texto adicional:

{{
  "score": <número del 1 al 10>,
  "level_detected": "<junior|mid|senior>",
  "strengths": "<fortalezas en 1 oración>",
  "improvements": "<qué mejorar en 1 oración>",
  "feedback": "<retroalimentación en 2-3 oraciones>"
}}"""


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Run:
    model: str
    case_id: str
    topic: str
    label: str          # completa / parcial / incorrecta
    expected_min: int
    expected_max: int
    elapsed: float
    json_ok: bool
    score: int | None
    feedback_text: str = ""
    hallucination_terms: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def exact_ok(self) -> bool:
        if self.score is None:
            return False
        return self.expected_min <= self.score <= self.expected_max

    @property
    def hallucinated(self) -> bool:
        return len(self.hallucination_terms) > 0


@dataclass
class ModelStats:
    model: str
    runs: list[Run] = field(default_factory=list)

    def _valid(self):
        return [r for r in self.runs if r.json_ok and r.score is not None]

    @property
    def json_rate(self) -> float:
        if not self.runs: return 0.0
        return sum(1 for r in self.runs if r.json_ok) / len(self.runs) * 100

    @property
    def accuracy(self) -> float:
        v = self._valid()
        if not v: return 0.0
        return sum(1 for r in v if r.exact_ok) / len(v) * 100

    @property
    def hallucination_rate(self) -> float:
        v = self._valid()
        if not v: return 0.0
        return sum(1 for r in v if r.hallucinated) / len(v) * 100

    @property
    def avg_time(self) -> float:
        times = [r.elapsed for r in self.runs if r.elapsed > 0]
        return round(statistics.mean(times), 2) if times else 0.0

    @property
    def consistency(self) -> float:
        """Desviación estándar del score por caso (promedio global). Menor = más consistente."""
        deviations = []
        # agrupar por (case_id, label)
        groups: dict[str, list[int]] = {}
        for r in self._valid():
            key = f"{r.case_id}_{r.label}"
            groups.setdefault(key, []).append(r.score)
        for scores in groups.values():
            if len(scores) > 1:
                deviations.append(statistics.stdev(scores))
        return round(statistics.mean(deviations), 2) if deviations else 0.0

    @property
    def discrimination_ok(self) -> bool:
        """¿El modelo da mayor score a respuestas completas que a incorrectas?"""
        def avg(label):
            sc = [r.score for r in self._valid() if r.label == label and r.score is not None]
            return statistics.mean(sc) if sc else 0
        return avg("completa") > avg("incorrecta")

    @property
    def avg_by_label(self) -> dict:
        out = {}
        for label in ("completa", "parcial", "incorrecta"):
            sc = [r.score for r in self._valid() if r.label == label]
            out[label] = round(statistics.mean(sc), 1) if sc else "—"
        return out

    @property
    def overall_score(self) -> float:
        """Score compuesto para ranking (mayor es mejor)."""
        return (
            self.accuracy * 0.35
            + self.json_rate * 0.25
            + (100 - self.hallucination_rate) * 0.25
            + max(0, (60 - self.avg_time)) / 60 * 100 * 0.15
        )


# ─── Utilidades ───────────────────────────────────────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def available_models(wanted: list[str]) -> tuple[list[str], list[str]]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        installed = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        installed = []
    ok, missing = [], []
    for m in wanted:
        base = m.split(":")[0]
        if any(x == m or x.startswith(base) for x in installed):
            ok.append(m)
        else:
            missing.append(m)
    return ok, missing


def call_model(model: str, prompt: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 600, "num_ctx": 2048},
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"], round(time.time() - t0, 2)


def parse_json(text: str) -> dict | None:
    for fn in (
        lambda: json.loads(text.strip()),
        lambda: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0)),
    ):
        try:
            return fn()
        except Exception:
            pass
    return None


def detect_hallucinations(feedback: str, forbidden: list[str], context_words: set[str]) -> list[str]:
    """
    Detecta términos que aparecen en el feedback pero:
      - están en la lista de términos prohibidos (tecnologías no relacionadas), O
      - son términos técnicos específicos que no aparecen en ningún input del caso
    """
    found = []
    fb_lower = feedback.lower()
    for term in forbidden:
        if term.lower() in fb_lower:
            found.append(term)
    return found


def run_case(model: str, case: dict, answer: dict, run_n: int) -> Run:
    prompt = PROMPT.format(
        question=case["question"],
        ideal_answer=case["ideal_answer"],
        keywords=", ".join(case["keywords"]),
        user_answer=answer["text"],
    )

    # Palabras del contexto que el modelo SÍ puede usar
    context = set(
        (case["question"] + case["ideal_answer"] + answer["text"]).lower().split()
    )

    try:
        raw, elapsed = call_model(model, prompt)
        data = parse_json(raw)

        if not data:
            return Run(model, case["id"], case["topic"], answer["label"],
                       answer["expected"][0], answer["expected"][1],
                       elapsed, False, None, error="JSON inválido")

        score = max(1, min(10, int(data.get("score", 0))))
        feedback = (
            data.get("feedback", "") + " " +
            data.get("strengths", "") + " " +
            data.get("improvements", "")
        )
        hallucinations = detect_hallucinations(feedback, case["forbidden_terms"], context)

        return Run(
            model=model,
            case_id=case["id"],
            topic=case["topic"],
            label=answer["label"],
            expected_min=answer["expected"][0],
            expected_max=answer["expected"][1],
            elapsed=elapsed,
            json_ok=True,
            score=score,
            feedback_text=feedback[:300],
            hallucination_terms=hallucinations,
        )

    except requests.exceptions.Timeout:
        return Run(model, case["id"], case["topic"], answer["label"],
                   answer["expected"][0], answer["expected"][1],
                   TIMEOUT, False, None, error="Timeout")
    except Exception as e:
        return Run(model, case["id"], case["topic"], answer["label"],
                   answer["expected"][0], answer["expected"][1],
                   0, False, None, error=str(e)[:80])


# ─── Reporte en consola ───────────────────────────────────────────────────────

W = 76

def sep(c="─"): print(c * W)

def print_report(stats: list[ModelStats]):
    ranked = sorted(stats, key=lambda s: s.overall_score, reverse=True)
    medals = ["🥇", "🥈", "🥉"] + ["  "] * 10

    sep("═")
    print(f"  BENCHMARK COMPARATIVO DE MODELOS LLM — Tech Interview AI")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sep("═")

    # ── Tabla de métricas principales ──
    print(f"\n  {'Modelo':<22} {'Exactitud':>10} {'Alucinac.':>10} {'Consistencia':>13} {'Velocidad':>10} {'JSON OK':>8}")
    sep()
    for i, s in enumerate(ranked):
        cons = f"σ={s.consistency}"
        print(
            f"  {medals[i]} {s.model:<20}"
            f"  {s.accuracy:>7.1f}%"
            f"  {s.hallucination_rate:>7.1f}%"
            f"  {cons:>13}"
            f"  {s.avg_time:>8.1f}s"
            f"  {s.json_rate:>6.0f}%"
        )

    # ── Discriminación por calidad de respuesta ──
    print(f"\n  Scores promedio por calidad de respuesta (esperado: completa > parcial > incorrecta)")
    sep()
    print(f"  {'Modelo':<22} {'Completa':>10} {'Parcial':>10} {'Incorrecta':>11} {'Discrimina':>11}")
    sep()
    for s in ranked:
        bl = s.avg_by_label
        disc = "✅ SÍ" if s.discrimination_ok else "❌ NO"
        print(
            f"  {s.model:<22}"
            f"  {str(bl['completa']):>10}"
            f"  {str(bl['parcial']):>10}"
            f"  {str(bl['incorrecta']):>11}"
            f"  {disc:>11}"
        )

    # ── Alucinaciones detectadas (detalle) ──
    halluc_runs = [r for s in stats for r in s.runs if r.hallucinated]
    if halluc_runs:
        print(f"\n  Alucinaciones detectadas (términos no relacionados en el feedback)")
        sep()
        for r in halluc_runs:
            terms = ", ".join(r.hallucination_terms)
            print(f"  {r.model:<22} [{r.case_id}/{r.label}] → {terms}")

    # ── Errores ──
    errors = [r for s in stats for r in s.runs if r.error]
    if errors:
        print(f"\n  Errores")
        sep()
        for r in errors:
            print(f"  {r.model:<22} [{r.case_id}/{r.label}] {r.error}")

    # ── Score compuesto y recomendación ──
    best = ranked[0]
    print()
    sep("═")
    print(f"\n  RESULTADO: Modelo recomendado → {best.model}")
    print(f"\n  Scores compuestos (exactitud×0.35 + JSON×0.25 + (1-alucinac.)×0.25 + velocidad×0.15):")
    for i, s in enumerate(ranked):
        bar = "█" * int(s.overall_score / 5)
        print(f"  {medals[i]} {s.model:<22} {s.overall_score:5.1f}/100  {bar}")

    print(f"\n  Para usar {best.model} en el proyecto:")
    print(f'  Edita .env → OLLAMA_MODEL={best.model}')
    sep("═")
    print()


# ─── Exportar CSV ─────────────────────────────────────────────────────────────

def export_csv(stats: list[ModelStats], path: Path):
    rows = []
    for s in stats:
        for r in s.runs:
            rows.append({
                "modelo": r.model,
                "caso": r.case_id,
                "tema": r.topic,
                "calidad_respuesta": r.label,
                "score_esperado_min": r.expected_min,
                "score_esperado_max": r.expected_max,
                "score_obtenido": r.score if r.score is not None else "",
                "exactitud_ok": int(r.exact_ok),
                "alucinacion": int(r.hallucinated),
                "terminos_alucinados": "|".join(r.hallucination_terms),
                "tiempo_seg": r.elapsed,
                "json_ok": int(r.json_ok),
                "error": r.error,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV exportado → {path}")


def export_json_summary(stats: list[ModelStats], path: Path):
    summary = []
    for s in stats:
        summary.append({
            "modelo": s.model,
            "exactitud_pct": round(s.accuracy, 1),
            "alucinacion_pct": round(s.hallucination_rate, 1),
            "consistencia_sigma": s.consistency,
            "velocidad_avg_seg": s.avg_time,
            "json_valido_pct": round(s.json_rate, 1),
            "discrimina_correctamente": s.discrimination_ok,
            "score_compuesto": round(s.overall_score, 1),
            "avg_por_calidad": s.avg_by_label,
        })
    summary.sort(key=lambda x: x["score_compuesto"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  JSON exportado → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM — Tech Interview AI")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, metavar="MODEL")
    parser.add_argument("--runs", type=int, default=RUNS, metavar="N",
                        help=f"Repeticiones por caso (default: {RUNS})")
    parser.add_argument("--export", action="store_true",
                        help="Exportar resultados a CSV y JSON")
    args = parser.parse_args()

    # ── Verificar Ollama ──
    if not check_ollama():
        print("❌ Ollama no está corriendo. Ejecuta: ollama serve")
        return

    # ── Verificar modelos ──
    ok_models, missing = available_models(args.models)
    if missing:
        print(f"\n⚠️  Modelos no descargados: {', '.join(missing)}")
        print("   Descárgalos con:")
        for m in missing:
            print(f"     ollama pull {m}")
        if not ok_models:
            return
        print(f"\n   Continuando con: {', '.join(ok_models)}\n")
    models = ok_models

    # ── Info del benchmark ──
    total_calls = len(models) * len(TEST_CASES) * 3 * args.runs
    print(f"\n{'═'*W}")
    print(f"  BENCHMARK — {len(models)} modelos · {len(TEST_CASES)} casos · "
          f"3 calidades × {args.runs} run(s) = {total_calls} llamadas")
    print(f"{'═'*W}\n")
    print(f"  Métricas: Exactitud · Alucinaciones · Consistencia · Velocidad · JSON válido\n")
    sep()

    # ── Ejecutar ──
    all_stats: dict[str, ModelStats] = {m: ModelStats(m) for m in models}
    done = 0

    for model in models:
        print(f"\n  Modelo: {model}")
        sep()
        for case in TEST_CASES:
            for answer in case["answers"]:
                for run_n in range(args.runs):
                    done += 1
                    tag = f"[{done}/{total_calls}] {case['id']} | {answer['label']:<10} run {run_n+1}"
                    print(f"  {tag}", end=" ... ", flush=True)

                    result = run_case(model, case, answer, run_n)
                    all_stats[model].runs.append(result)

                    if result.json_ok:
                        acc = "✅" if result.exact_ok else "⚠️ "
                        hall = f" 🔴alucinación:{result.hallucination_terms}" if result.hallucinated else ""
                        print(f"score={result.score} {acc} ({result.elapsed}s){hall}")
                    else:
                        print(f"❌ {result.error} ({result.elapsed}s)")

    # ── Reporte ──
    print()
    print_report(list(all_stats.values()))

    # ── Exportar ──
    if args.export:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("benchmark_results")
        out_dir.mkdir(exist_ok=True)
        export_csv(list(all_stats.values()), out_dir / f"benchmark_{ts}.csv")
        export_json_summary(list(all_stats.values()), out_dir / f"benchmark_{ts}.json")
        print()


if __name__ == "__main__":
    main()
