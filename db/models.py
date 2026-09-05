from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ReconciliationRun(Base):
    __tablename__ = "reconciliation_runs"

    run_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scenario = Column(String(32), default="default")
    total_erp_count = Column(Integer, default=0)
    total_bank_count = Column(Integer, default=0)
    total_gw_count = Column(Integer, default=0)
    matched_count = Column(Integer, default=0)
    review_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    unmatched_count = Column(Integer, default=0)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    chain_verified = Column(Boolean, default=True)

    decisions = relationship("MatchDecision", back_populates="run", cascade="all, delete-orphan")
    exceptions = relationship("ExceptionRecord", back_populates="run", cascade="all, delete-orphan")


class MatchDecision(Base):
    __tablename__ = "match_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("reconciliation_runs.run_id"), nullable=False)
    erp_invoice_id = Column(String(64), index=True)
    bank_ref = Column(String(64), nullable=True)
    gw_ref = Column(String(64), nullable=True)
    erp_amount = Column(Float, default=0.0)
    bank_amount = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    confidence_zone = Column(String(16), default="LOW")
    final_decision = Column(String(32), default="UNMATCHED")
    match_type = Column(String(32), default="ONE_TO_ONE")
    audit_risk = Column(String(16), default="LOW")
    auditor_reason = Column(Text, nullable=True)
    this_hash = Column(String(64), nullable=True)
    previous_hash = Column(String(64), nullable=True)

    run = relationship("ReconciliationRun", back_populates="decisions")


class ExceptionRecord(Base):
    __tablename__ = "exception_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("reconciliation_runs.run_id"), nullable=False)
    erp_invoice_id = Column(String(64), index=True)
    amount = Column(Float, default=0.0)
    category = Column(String(64), default="OTHER")
    explanation = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    severity = Column(String(16), default="MEDIUM")

    run = relationship("ReconciliationRun", back_populates="exceptions")
