import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from app.core.logging import setup_logging
setup_logging()

log = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from services.questions import (
    get_available_categories,
    get_interview_questions,
    get_question_by_id,
    get_stats,
    TOTAL_QUESTIONS,
)
from services.generator import generate_question
from app.ai.evaluator import evaluate_answer
from app.ai.ollama_client import ollama_health_check
from app.db.database import get_db, create_tables
from app.db.crud import (
    create_user, get_user_by_username, get_user_by_email,
    save_evaluation, get_evaluation_by_id, get_user_evaluations,
    save_generated_question,
)
from app.auth.auth import hash_password, authenticate_user

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando Tech Interview AI...")
    try:
        create_tables()
        log.info("Tablas de base de datos verificadas.")
    except Exception as e:
        log.error("No se pudo conectar a la base de datos: %s", e)

    import asyncio
    from app.ai.embeddings import _load_model
    try:
        await asyncio.get_event_loop().run_in_executor(None, _load_model)
        log.info("Modelo de embeddings listo.")
    except Exception as e:
        log.warning("No se pudo pre-cargar embeddings: %s", e)

    yield
    log.info("Apagando Tech Interview AI.")


app = FastAPI(title="Tech Interview AI", version="1.0", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=7 * 24 * 3600)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app/templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), name="static")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def session_user(request: Request) -> dict | None:
    uid = request.session.get("user_id")
    return {"id": uid, "username": request.session.get("username")} if uid else None


def _detect_final_level(questions: list, results: list) -> dict:
    level_scores: dict[str, list] = {"junior": [], "mid": [], "senior": []}
    for i, result in enumerate(results):
        if i < len(questions):
            diff = (questions[i] or {}).get("difficulty", "mid")
            if diff in level_scores:
                level_scores[diff].append(result.get("score", 0))

    level_avgs = {
        lvl: round(sum(v) / len(v), 1) if v else 0.0
        for lvl, v in level_scores.items()
    }

    if level_avgs["senior"] >= 7:
        final_level, level_color = "Senior", "danger"
    elif level_avgs["mid"] >= 7:
        final_level, level_color = "Mid", "warning"
    elif level_avgs["junior"] >= 5:
        final_level, level_color = "Junior", "success"
    else:
        final_level, level_color = "En desarrollo", "muted"

    scores = [r.get("score", 0) for r in results]
    overall_avg = round(sum(scores) / len(scores), 1) if scores else 0.0
    qualities = [r.get("quality_percent", 0) for r in results]
    avg_quality = round(sum(qualities) / len(qualities), 1) if qualities else 0.0

    return {
        "level_avgs": level_avgs,
        "final_level": final_level,
        "level_color": level_color,
        "overall_avg": overall_avg,
        "avg_quality": avg_quality,
    }


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if session_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...),
                     db: Session = Depends(get_db)):
    user = authenticate_user(db, username, password)
    if not user:
        return templates.TemplateResponse(request=request, name="login.html",
                                          context={"error": "Usuario o contraseña incorrectos."})
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    return RedirectResponse("/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if session_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="register.html", context={"error": None})


@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request, username: str = Form(...), email: str = Form(...),
                        password: str = Form(...), db: Session = Depends(get_db)):
    error = None
    if len(username) < 3:
        error = "El nombre de usuario debe tener al menos 3 caracteres."
    elif len(password) < 6:
        error = "La contraseña debe tener al menos 6 caracteres."
    elif get_user_by_username(db, username):
        error = "Ese nombre de usuario ya está en uso."
    elif get_user_by_email(db, email):
        error = "Ya existe una cuenta con ese correo."
    if error:
        return templates.TemplateResponse(request=request, name="register.html", context={"error": error})
    user = create_user(db, username=username, email=email, password_hash=hash_password(password))
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    return RedirectResponse("/", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ─── Home ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    interview = request.session.get("interview")
    if interview and not interview.get("done"):
        return RedirectResponse("/question", status_code=302)
    health = ollama_health_check()
    return templates.TemplateResponse(request=request, name="index.html", context={
        "categories": get_available_categories(),
        "ai_ok": health["ok"],
        "ai_warning": health.get("warning"),
        "ai_model": health.get("active_model", ""),
        "current_user": user,
    })


# ─── Entrevista ───────────────────────────────────────────────────────────────

@app.post("/interview/start")
async def interview_start(request: Request,
                          category: str = Form(default="all")):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    cat = category if category != "all" else None
    question_ids = get_interview_questions(cat)

    request.session["interview"] = {
        "category": category,
        "question_ids": question_ids,
        "current_idx": 0,
        "results": [],
        "done": False,
    }
    log.info("Entrevista iniciada — usuario=%s categoría=%s", user["username"], category)
    return RedirectResponse("/question", status_code=302)


@app.get("/question", response_class=HTMLResponse)
async def question_page(request: Request):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    interview = request.session.get("interview")
    if not interview:
        return RedirectResponse("/", status_code=302)

    idx = interview["current_idx"]
    question_ids = interview["question_ids"]

    if idx >= len(question_ids):
        return RedirectResponse("/summary", status_code=302)

    question = get_question_by_id(question_ids[idx])
    if not question:
        return RedirectResponse("/", status_code=302)

    health = ollama_health_check()
    return templates.TemplateResponse(request=request, name="question.html", context={
        "question": question,
        "progress_current": idx + 1,
        "progress_total": TOTAL_QUESTIONS,
        "progress_pct": round((idx / TOTAL_QUESTIONS) * 100),
        "ai_ok": health["ok"],
        "ai_warning": health.get("warning"),
        "current_user": user,
    })


@app.post("/evaluate", response_class=HTMLResponse)
def evaluate(request: Request,
             question: str = Form(...),
             ideal_answer: str = Form(...),
             user_answer: str = Form(...),
             keywords: str = Form(default=""),
             question_id: str = Form(default=""),
             difficulty: str = Form(default=""),
             category: str = Form(default=""),
             code_snippet: str = Form(default=""),
             from_interview: str = Form(default="0"),
             db: Session = Depends(get_db)):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    interview = request.session.get("interview")
    in_interview = from_interview == "1" and bool(interview) and not interview.get("done")

    if not user_answer.strip():
        ctx = {"error": "Por favor escribe una respuesta antes de evaluar.",
               "ai_ok": True, "ai_warning": None, "current_user": user}
        if in_interview:
            idx = interview["current_idx"]
            ids = interview["question_ids"]
            ctx.update({
                "question": get_question_by_id(ids[idx]),
                "progress_current": idx + 1,
                "progress_total": TOTAL_QUESTIONS,
                "progress_pct": round((idx / TOTAL_QUESTIONS) * 100),
            })
            return templates.TemplateResponse(request=request, name="question.html", context=ctx)
        return RedirectResponse("/", status_code=302)

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    try:
        result = evaluate_answer(
            question=question, ideal_answer=ideal_answer,
            user_answer=user_answer, keywords=kw_list,
            question_id=int(question_id) if question_id.isdigit() else None,
            code_snippet=code_snippet or None,
        )
        result.update({
            "question_text": question,
            "user_answer_text": user_answer,
            "ideal_answer_text": ideal_answer,
            "difficulty": difficulty,
            "category": category,
        })
    except RuntimeError as e:
        result = {"error": str(e)}

    # ── Flujo entrevista: guardar y pasar a la siguiente pregunta ──
    if in_interview:
        eval_id = None
        if "error" not in result:
            try:
                ev = save_evaluation(db, user["id"], result)
                eval_id = ev.id
            except Exception as e:
                log.error("Error guardando evaluación: %s", e)

        idx = interview["current_idx"]
        results = interview.get("results", [])
        compact = {
            "score": result.get("score", 0),
            "quality_label": result.get("quality_label", ""),
            "quality_percent": result.get("quality_percent", 0),
            "difficulty": difficulty,
            "category": category,
            "eval_id": eval_id,
            "keywords_found": result.get("keywords_found", []),
            "keywords_missing": result.get("keywords_missing", []),
        }
        if len(results) <= idx:
            results.append(compact)
        else:
            results[idx] = compact
        interview["results"] = results
        request.session["interview"] = interview
        # Redirige directo a la siguiente pregunta, sin mostrar resultado
        return RedirectResponse("/next", status_code=302)

    # ── Fuera de entrevista (página Generar): sí mostrar resultado ──
    try:
        save_evaluation(db, user["id"], result)
    except Exception as e:
        log.error("Error guardando evaluación: %s", e)

    return templates.TemplateResponse(request=request, name="result.html", context={
        "result": result,
        "in_interview": False,
        "is_last": False,
        "current_user": user,
    })


@app.get("/next")
async def next_question(request: Request):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    interview = request.session.get("interview")
    if not interview:
        return RedirectResponse("/", status_code=302)

    idx = interview["current_idx"]
    if len(interview.get("results", [])) <= idx:
        return RedirectResponse("/question", status_code=302)

    interview["current_idx"] = idx + 1
    if interview["current_idx"] >= len(interview["question_ids"]):
        interview["done"] = True
    request.session["interview"] = interview

    return RedirectResponse("/summary" if interview["done"] else "/question", status_code=302)


@app.get("/summary", response_class=HTMLResponse)
async def summary(request: Request, db: Session = Depends(get_db)):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    interview = request.session.get("interview")
    if not interview or not interview.get("results"):
        return RedirectResponse("/", status_code=302)

    question_ids = interview["question_ids"]
    results = interview["results"]
    questions = [get_question_by_id(qid) for qid in question_ids]
    stats = _detect_final_level(questions, results)

    detail = []
    for i, (q, r) in enumerate(zip(questions, results)):
        ev = get_evaluation_by_id(db, r["eval_id"]) if r.get("eval_id") else None
        detail.append({
            "num": i + 1,
            "question": (q or {}).get("question", ""),
            "difficulty": r.get("difficulty") or (q or {}).get("difficulty", ""),
            "category": r.get("category") or (q or {}).get("category", ""),
            "score": r.get("score", 0),
            "quality_label": r.get("quality_label", ""),
            "quality_percent": r.get("quality_percent", 0),
            "feedback": ev.feedback if ev else "",
            "strengths": ev.strengths if ev else "",
            "improvements": ev.improvements if ev else "",
            "user_answer": ev.user_answer if ev else "",
            "ideal_answer": ev.ideal_answer_text if ev else "",
            "keywords_found": r.get("keywords_found", []),
            "keywords_missing": r.get("keywords_missing", []),
        })

    interview["done"] = True
    request.session["interview"] = interview

    return templates.TemplateResponse(request=request, name="summary.html", context={
        "detail": detail,
        "stats": stats,
        "current_user": user,
    })


@app.get("/interview/reset")
async def reset_interview(request: Request):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    request.session.pop("interview", None)
    return RedirectResponse("/", status_code=302)


# ─── Historial ────────────────────────────────────────────────────────────────

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(get_db)):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    evaluations = get_user_evaluations(db, user["id"])
    return templates.TemplateResponse(request=request, name="history.html", context={
        "current_user": user,
        "evaluations": evaluations,
    })


# ─── Generar pregunta ─────────────────────────────────────────────────────────

@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request, category: str = "backend", difficulty: str = "mid"):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request=request, name="generate.html", context={
        "categories": get_available_categories(),
        "levels": ["junior", "mid", "senior"],
        "selected_category": category,
        "selected_level": difficulty,
        "current_user": user,
    })


@app.post("/generate", response_class=HTMLResponse)
def generate_post(request: Request, category: str = Form(default="backend"),
                  difficulty: str = Form(default="mid"), db: Session = Depends(get_db)):
    user = session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    generated = None
    error = None
    try:
        generated = generate_question(category, difficulty)
        try:
            save_generated_question(db, user["id"], generated)
        except Exception as e:
            log.error("Error guardando pregunta generada: %s", e)
    except RuntimeError as e:
        error = str(e)
    return templates.TemplateResponse(request=request, name="generate.html", context={
        "categories": get_available_categories(),
        "levels": ["junior", "mid", "senior"],
        "selected_category": category,
        "selected_level": difficulty,
        "generated": generated,
        "error": error,
        "current_user": user,
    })


# ─── Utilidades ───────────────────────────────────────────────────────────────

@app.get("/stats")
async def stats():
    return JSONResponse(get_stats())


@app.get("/health")
async def health():
    return JSONResponse(ollama_health_check())
