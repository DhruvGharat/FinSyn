import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, ReconciliationRun, MatchDecision, ExceptionRecord

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/reconciliation.db")

# Ensure directory exists for SQLite
if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_reconciliation_run(run_data: dict, decisions: list, exceptions: list) -> str:
    db = SessionLocal()
    try:
        run_obj = ReconciliationRun(
            run_id=run_data["run_id"],
            scenario=run_data.get("scenario", "default"),
            total_erp_count=run_data.get("total_erp_count", 0),
            total_bank_count=run_data.get("total_bank_count", 0),
            total_gw_count=run_data.get("total_gw_count", 0),
            matched_count=run_data.get("matched_count", 0),
            review_count=run_data.get("review_count", 0),
            rejected_count=run_data.get("rejected_count", 0),
            unmatched_count=run_data.get("unmatched_count", 0),
            precision=run_data.get("precision"),
            recall=run_data.get("recall"),
            f1_score=run_data.get("f1_score"),
            chain_verified=run_data.get("chain_verified", True)
        )
        db.add(run_obj)

        for d in decisions:
            audit_meta = d.get("audit_metadata", {})
            decision_obj = MatchDecision(
                run_id=run_data["run_id"],
                erp_invoice_id=d.get("erp_invoice_id"),
                bank_ref=d.get("bank_ref"),
                gw_ref=d.get("gw_ref"),
                erp_amount=float(d.get("erp_amount", 0.0)),
                bank_amount=float(d.get("bank_amount", 0.0)),
                confidence=float(d.get("confidence", 0.0)),
                confidence_zone=d.get("confidence_zone", "LOW"),
                final_decision=d.get("final_decision", "UNMATCHED"),
                match_type=d.get("match_type", "ONE_TO_ONE"),
                audit_risk=audit_meta.get("audit_risk", "LOW"),
                auditor_reason=audit_meta.get("auditor_reason", ""),
                this_hash=d.get("this_hash"),
                previous_hash=d.get("previous_hash")
            )
            db.add(decision_obj)

        for e in exceptions:
            exc_obj = ExceptionRecord(
                run_id=run_data["run_id"],
                erp_invoice_id=e.get("invoice_id"),
                amount=float(e.get("amount", 0.0)),
                category=e.get("exception_category", "OTHER"),
                explanation=e.get("explanation", ""),
                recommended_action=e.get("recommended_action", ""),
                severity=e.get("severity", "MEDIUM")
            )
            db.add(exc_obj)

        db.commit()
        return run_data["run_id"]
    except Exception as ex:
        db.rollback()
        raise ex
    finally:
        db.close()
