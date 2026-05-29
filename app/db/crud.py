from __future__ import annotations
from sqlalchemy.orm import Session
from app.db.models import User, Evaluation, GeneratedQuestion


def create_user(db: Session, username: str, email: str, password_hash: str) -> User:
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def save_evaluation(db: Session, user_id: int, data: dict) -> Evaluation:
    ev = Evaluation(
        user_id=user_id,
        question_text=data.get("question_text", ""),
        ideal_answer_text=data.get("ideal_answer_text", ""),
        user_answer=data.get("user_answer_text", ""),
        score=data.get("score"),
        quality_percent=data.get("quality_percent"),
        quality_label=data.get("quality_label"),
        category=data.get("category"),
        difficulty=data.get("difficulty"),
        level_detected=data.get("level_detected"),
        feedback=data.get("feedback"),
        strengths=data.get("strengths"),
        improvements=data.get("improvements"),
        semantic_similarity=data.get("semantic_similarity"),
        keyword_coverage=data.get("keyword_coverage"),
        response_time_seconds=data.get("response_time_seconds"),
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


def get_evaluation_by_id(db: Session, eval_id: int) -> Evaluation | None:
    return db.query(Evaluation).filter(Evaluation.id == eval_id).first()


def get_user_evaluations(db: Session, user_id: int, limit: int = 50) -> list[Evaluation]:
    return (
        db.query(Evaluation)
        .filter(Evaluation.user_id == user_id)
        .order_by(Evaluation.created_at.desc())
        .limit(limit)
        .all()
    )


def save_generated_question(db: Session, user_id: int, data: dict) -> GeneratedQuestion:
    gq = GeneratedQuestion(
        user_id=user_id,
        question_text=data.get("question", ""),
        ideal_answer=data.get("ideal_answer", ""),
        keywords=data.get("keywords", []),
        difficulty=data.get("difficulty"),
        category=data.get("category"),
    )
    db.add(gq)
    db.commit()
    db.refresh(gq)
    return gq
