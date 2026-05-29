from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluations = relationship("Evaluation", back_populates="user", cascade="all, delete-orphan")
    generated_questions = relationship("GeneratedQuestion", back_populates="user", cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    ideal_answer_text = Column(Text)
    user_answer = Column(Text, nullable=False)
    score = Column(Integer)
    quality_percent = Column(Float)
    quality_label = Column(String(20))
    category = Column(String(50))
    difficulty = Column(String(20))
    level_detected = Column(String(20))
    feedback = Column(Text)
    strengths = Column(Text)
    improvements = Column(Text)
    semantic_similarity = Column(Float)
    keyword_coverage = Column(Float)
    response_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="evaluations")


class GeneratedQuestion(Base):
    __tablename__ = "generated_questions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    question_text = Column(Text, nullable=False)
    ideal_answer = Column(Text)
    keywords = Column(JSON)
    difficulty = Column(String(20))
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="generated_questions")
