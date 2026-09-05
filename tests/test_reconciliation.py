import os
import json
import pandas as pd

from datetime import datetime
from fastapi.testclient import TestClient

from matcher.engine import parse_date, amount_similarity, date_similarity, reference_similarity, compute_pair_cost, generate_candidate_pairs, normalize_ledger_columns
from matcher.hungarian import run_hungarian, evaluate_against_ground_truth
from matcher.features import engineer_features, label_candidates
from matcher.scorer import train_scorer, score_candidates
from audit.hash_chain import AuditChain, AuditRecord
from generator.synthetic import generate_transaction_set
from matcher.pipeline import run_reconciliation_pipeline
from api.main import app


# 1. Date parsing test
def test_date_parsing():
    assert parse_date("2026-03-15") == datetime(2026, 3, 15)
    assert parse_date("15/03/2026") == datetime(2026, 3, 15)
    assert parse_date("invalid") is None
    assert parse_date(None) is None


# 2. Amount normalization test
def test_amount_normalization():
    assert amount_similarity(100.0, 100.0) == 0.0
    assert amount_similarity(100.0, 101.0) < 0.05
    assert amount_similarity(100.0, 200.0) == 1.0


# 3. Candidate generation test
def test_candidate_generation():
    erp_df = pd.DataFrame([{"invoice_id": "INV-001", "amount": 1000.0, "date": "2026-03-01", "currency": "INR"}])
    bank_df = pd.DataFrame([{"bank_ref": "BANK-001", "amount": 1000.0, "date": "2026-03-02", "currency": "INR"}])
    cands = generate_candidate_pairs(erp_df, bank_df)
    assert len(cands) == 1
    assert cands.iloc[0]["cost"] < 0.5


# 4. Cost calculation test
def test_cost_calculation():
    cost = compute_pair_cost(1000.0, 1000.0, datetime(2026, 3, 1), datetime(2026, 3, 1), "INV-1", "INV-1")
    assert cost == 0.0


# 5. Hungarian matching test
def test_hungarian_matching():
    erp_df = pd.DataFrame([{"invoice_id": "INV-001", "amount": 1000.0, "date": "2026-03-01"}])
    bank_df = pd.DataFrame([{"bank_ref": "BANK-001", "amount": 1000.0, "date": "2026-03-01"}])
    cands = generate_candidate_pairs(erp_df, bank_df)
    assign = run_hungarian(erp_df, bank_df, cands)
    assert len(assign[assign["decision"] == "MATCHED"]) == 1


# 6. Unequal row counts test
def test_unequal_row_counts():
    erp_df = pd.DataFrame([
        {"invoice_id": "INV-001", "amount": 1000.0, "date": "2026-03-01"},
        {"invoice_id": "INV-002", "amount": 2000.0, "date": "2026-03-01"}
    ])
    bank_df = pd.DataFrame([{"bank_ref": "BANK-001", "amount": 1000.0, "date": "2026-03-01"}])
    cands = generate_candidate_pairs(erp_df, bank_df)
    assign = run_hungarian(erp_df, bank_df, cands)
    assert len(assign) == 2
    assert len(assign[assign["decision"] == "MATCHED"]) == 1


# 7. XGBoost feature generation test
def test_feature_generation():
    cands = pd.DataFrame([{
        "erp_invoice_id": "INV-1", "bank_ref": "B-1", "gw_ref": "GW-1",
        "erp_amount": 1000.0, "bank_amount": 1000.0,
        "erp_date": "2026-03-01", "bank_date": "2026-03-01", "cost": 0.0
    }])
    feats = engineer_features(cands)
    assert len(feats) == 1
    assert "amt_diff_abs" in feats.columns


# 8. XGBoost inference test
def test_xgboost_inference():
    cands = pd.DataFrame([
        {
            "erp_invoice_id": "INV-1", "bank_ref": "B-1", "gw_ref": "GW-1",
            "erp_amount": 1000.0, "bank_amount": 1000.0,
            "erp_date": "2026-03-01", "bank_date": "2026-03-01", "cost": 0.0
        },
        {
            "erp_invoice_id": "INV-2", "bank_ref": "B-99", "gw_ref": "GW-99",
            "erp_amount": 1000.0, "bank_amount": 5000.0,
            "erp_date": "2026-03-01", "bank_date": "2026-03-25", "cost": 0.9
        }
    ])
    feats = engineer_features(cands)
    feats["label"] = [1, 0]
    model, _ = train_scorer(feats)
    scored = score_candidates(model, feats)
    assert "confidence" in scored.columns
    assert len(scored) == 2



# 9. Ground-truth evaluation test
def test_ground_truth_evaluation():
    assignments = pd.DataFrame([{"erp_invoice_id": "INV-1", "bank_ref": "B-1", "decision": "MATCHED"}])
    gt = {"INV-1": {"bank_refs": ["B-1"]}}
    eval_res = evaluate_against_ground_truth(assignments, gt)
    assert eval_res["precision"] == 1.0
    assert eval_res["recall"] == 1.0


# 10. Exception classification test
def test_exception_classification():
    from agents.nodes import deterministic_fallback_explainer
    exc = {"invoice_id": "INV-1", "amount": 1000.0, "closest_bank_amount": 1005.0}
    exp = deterministic_fallback_explainer(exc)
    assert exp["exception_category"] == "FX_ROUNDING"


# 11. Audit hash generation test
def test_audit_hash_generation():
    chain = AuditChain()
    rec = chain.add_record("INV-1", "B-1", "GW-1", 0.95, "AUTO_ACCEPTED")
    assert len(rec.this_hash) == 64


# 12. Audit chain verification test
def test_audit_chain_verification():
    chain = AuditChain()
    chain.add_record("INV-1", "B-1", "GW-1", 0.95, "AUTO_ACCEPTED")
    chain.add_record("INV-2", "B-2", "GW-2", 0.88, "AUTO_ACCEPTED")
    valid, idx = chain.verify_audit_chain()
    assert valid is True
    assert idx == -1


# 13. Tamper detection test
def test_tamper_detection():
    chain = AuditChain()
    chain.add_record("INV-1", "B-1", "GW-1", 0.95, "AUTO_ACCEPTED")
    chain.add_record("INV-2", "B-2", "GW-2", 0.88, "AUTO_ACCEPTED")
    # Mutate record 0
    chain.chain[0].confidence = 0.10
    valid, idx = chain.verify_audit_chain()
    assert valid is False
    assert idx == 0


# 14. CSV validation test
def test_csv_validation():
    erp = pd.DataFrame({"Invoice ID": ["INV-1"], "Amount": [100.0], "Date": ["2026-03-01"]})
    norm = normalize_ledger_columns(erp, "erp")
    assert "invoice_id" in norm.columns
    assert "amount" in norm.columns


# 15. API health endpoint test
def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# 16. Reconciliation API test
def test_api_reconciliation():
    client = TestClient(app)
    erp_csv = "invoice_id,amount,date\nINV-001,500.0,2026-03-01\n"
    bank_csv = "bank_ref,amount,date\nRZRPY-1,500.0,2026-03-01\n"

    response = client.post(
        "/reconcile",
        files={
            "erp_file": ("erp.csv", erp_csv.encode("utf-8"), "text/csv"),
            "bank_file": ("bank.csv", bank_csv.encode("utf-8"), "text/csv")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "run_id" in data


# 17. Full end-to-end pipeline test
def test_full_pipeline_end_to_end():
    erp_df, bank_df, gw_df, gt = generate_transaction_set(n_transactions=20, seed=42, scenario="default")
    res = run_reconciliation_pipeline(erp_df, bank_df, gw_df, gt)
    assert res["chain_verified"] is True
    assert len(res["decisions"]) > 0
