import uuid
import json
import pandas as pd
from datetime import datetime

from matcher.engine import normalize_ledger_columns, generate_candidate_pairs
from matcher.hungarian import run_hungarian, evaluate_against_ground_truth
from matcher.features import engineer_features, label_candidates
from matcher.scorer import load_or_train_scorer, train_scorer, score_candidates
from agents.nodes import run_exception_explainer, run_adversarial_auditor, run_assembler
from audit.hash_chain import AuditChain
from db.database import init_db, save_reconciliation_run

init_db()


def run_reconciliation_pipeline(
    erp_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    gw_df: pd.DataFrame = None,
    ground_truth: dict = None,
    scenario: str = "default"
) -> dict:
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"

    # 1. Normalize CSVs
    erp_df = normalize_ledger_columns(erp_df, "erp")
    bank_df = normalize_ledger_columns(bank_df, "bank")
    if gw_df is not None and not gw_df.empty:
        gw_df = normalize_ledger_columns(gw_df, "gateway")

    # 2. Candidate Generation
    candidates_df = generate_candidate_pairs(erp_df, bank_df, gw_df=gw_df)

    # 3. Hungarian & Split/Merge Matching
    hungarian_assignments = run_hungarian(erp_df, bank_df, candidates_df)

    # 4. XGBoost Feature Engineering & Scoring
    features_df = engineer_features(candidates_df)
    ml_eval_metrics = {}

    if ground_truth and not features_df.empty:
        features_df = label_candidates(features_df, ground_truth)
        model, ml_eval_metrics = train_scorer(features_df)
    else:
        model = load_or_train_scorer(features_df)

    if not features_df.empty:
        scored_df = score_candidates(model, features_df)
    else:
        scored_df = pd.DataFrame(columns=["erp_invoice_id", "bank_ref", "gw_ref", "confidence", "confidence_zone"])

    # Merge Hungarian assignment decision into scored candidate pairs
    hungarian_map = {row["erp_invoice_id"]: row for _, row in hungarian_assignments.iterrows()}

    high_pairs = []
    medium_pairs = []
    low_pairs = []

    for _, row in scored_df.iterrows():
        p_dict = row.to_dict()
        zone = p_dict.get("confidence_zone", "LOW")
        if zone == "HIGH":
            high_pairs.append(p_dict)
        elif zone == "MEDIUM":
            medium_pairs.append(p_dict)
        else:
            low_pairs.append(p_dict)

    # Identify Unmatched ERP rows
    matched_erp_ids = {p["erp_invoice_id"] for p in high_pairs + medium_pairs}
    exceptions = []

    for _, erp_row in erp_df.iterrows():
        eid = str(erp_row["invoice_id"])
        if eid not in matched_erp_ids:
            # Closest candidate for explanation context
            cands = scored_df[scored_df["erp_invoice_id"] == eid].sort_values("confidence", ascending=False) if not scored_df.empty else pd.DataFrame()
            closest = cands.iloc[0].to_dict() if not cands.empty else None

            exceptions.append({
                "invoice_id": eid,
                "amount": float(erp_row["amount"]),
                "date": str(erp_row["date"]),
                "customer_id": str(erp_row.get("customer_id", "N/A")),
                "description": str(erp_row.get("description", "N/A")),
                "closest_bank_ref": closest.get("bank_ref", "None") if closest else "None",
                "closest_bank_amount": closest.get("bank_amount", "N/A") if closest else "N/A",
                "closest_bank_date": closest.get("bank_date", "N/A") if closest else "N/A",
                "closest_cost": closest.get("combined_cost", "N/A") if closest else "N/A",
            })

    # 5. LangGraph Agent Nodes
    state = {
        "medium_pairs": medium_pairs,
        "high_pairs": high_pairs,
        "exceptions": exceptions,
        "explained": [],
        "audited": [],
        "final_matches": []
    }

    state = run_exception_explainer(state)
    state = run_adversarial_auditor(state)
    state = run_assembler(state)

    final_matches = state["final_matches"]
    explained_exceptions = state["explained"]

    # 6. Tamper-Evident SHA-256 Audit Chain
    audit_chain = AuditChain()
    processed_decisions = []

    for fm in final_matches:
        rec = audit_chain.add_record(
            erp_ref=fm.get("erp_invoice_id"),
            bank_ref=fm.get("bank_ref"),
            gw_ref=fm.get("gw_ref", "NONE"),
            confidence=fm.get("confidence", 1.0),
            decision=fm.get("final_decision", "UNMATCHED"),
            metadata=fm.get("audit_metadata", {})
        )
        processed_decisions.append({
            **fm,
            "this_hash": rec.this_hash,
            "previous_hash": rec.previous_hash
        })

    is_valid, corrupted_idx = audit_chain.verify_audit_chain()

    # Calculate overall ground truth evaluation if ground_truth is provided
    ground_truth_eval = {}
    if ground_truth:
        ground_truth_eval = evaluate_against_ground_truth(hungarian_assignments, ground_truth)

    # 7. Persist to DB
    run_summary = {
        "run_id": run_id,
        "scenario": scenario,
        "total_erp_count": len(erp_df),
        "total_bank_count": len(bank_df),
        "total_gw_count": len(gw_df) if gw_df is not None else 0,
        "matched_count": sum(1 for d in processed_decisions if d.get("final_decision") in ["AUTO_ACCEPTED", "REVIEW_APPROVED"]),
        "review_count": sum(1 for d in processed_decisions if d.get("final_decision") == "MANUAL_REVIEW"),
        "rejected_count": sum(1 for d in processed_decisions if d.get("final_decision") == "AUTO_REJECTED"),
        "unmatched_count": len(explained_exceptions),
        "precision": ground_truth_eval.get("precision"),
        "recall": ground_truth_eval.get("recall"),
        "f1_score": ground_truth_eval.get("f1_score"),
        "chain_verified": is_valid
    }

    try:
        save_reconciliation_run(run_summary, processed_decisions, explained_exceptions)
    except Exception:
        pass

    return {
        "run_id": run_id,
        "summary": run_summary,
        "decisions": processed_decisions,
        "exceptions": explained_exceptions,
        "audit_chain_df": audit_chain.to_dataframe(),
        "chain_verified": is_valid,
        "corrupted_index": corrupted_idx,
        "ground_truth_eval": ground_truth_eval,
        "ml_eval_metrics": ml_eval_metrics
    }
