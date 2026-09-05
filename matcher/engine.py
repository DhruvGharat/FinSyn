import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime
from itertools import product


def normalize_ledger_columns(df: pd.DataFrame, ledger_type: str) -> pd.DataFrame:
    """
    Standardize column names across user-uploaded CSVs.
    ledger_type: 'erp', 'bank', or 'gateway'
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Normalize header strings
    column_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        
        if ledger_type == "erp":
            if c_clean in ["invoice_id", "invoice", "inv_id", "doc_no", "id", "transaction_id"]:
                column_map[col] = "invoice_id"
            elif c_clean in ["amount", "amt", "total", "value", "inv_amount"]:
                column_map[col] = "amount"
            elif c_clean in ["date", "txn_date", "invoice_date", "created_at"]:
                column_map[col] = "date"
            elif c_clean in ["currency", "curr"]:
                column_map[col] = "currency"
            elif c_clean in ["customer_id", "customer", "client", "cust_id"]:
                column_map[col] = "customer_id"
            elif c_clean in ["description", "desc", "narration", "memo"]:
                column_map[col] = "description"

        elif ledger_type == "bank":
            if c_clean in ["bank_ref", "reference", "ref_no", "utr", "txn_id", "id", "bank_id"]:
                column_map[col] = "bank_ref"
            elif c_clean in ["amount", "amt", "credit", "deposit_amount"]:
                column_map[col] = "amount"
            elif c_clean in ["date", "txn_date", "value_date", "post_date"]:
                column_map[col] = "date"
            elif c_clean in ["currency", "curr"]:
                column_map[col] = "currency"
            elif c_clean in ["description", "desc", "particulars", "narration"]:
                column_map[col] = "description"
            elif c_clean in ["type", "txn_type", "dr_cr"]:
                column_map[col] = "type"

        elif ledger_type == "gateway":
            if c_clean in ["gw_ref", "gateway_ref", "payment_id", "settlement_id", "razorpay_id", "id"]:
                column_map[col] = "gw_ref"
            elif c_clean in ["amount", "amt", "settlement_amount"]:
                column_map[col] = "amount"
            elif c_clean in ["fee", "fees", "charges", "gateway_fee"]:
                column_map[col] = "fee"
            elif c_clean in ["date", "settlement_date", "txn_date"]:
                column_map[col] = "date"
            elif c_clean in ["currency", "curr"]:
                column_map[col] = "currency"
            elif c_clean in ["erp_ref_hint", "order_id", "invoice_hint", "notes"]:
                column_map[col] = "erp_ref_hint"

    df = df.rename(columns=column_map)

    # Supply default fallback values for mandatory columns if missing
    if ledger_type == "erp":
        if "invoice_id" not in df.columns: df["invoice_id"] = [f"INV-{i+1:05d}" for i in range(len(df))]
        if "amount" not in df.columns: df["amount"] = 0.0
        if "date" not in df.columns: df["date"] = datetime.now().strftime("%Y-%m-%d")
        if "currency" not in df.columns: df["currency"] = "INR"
        if "description" not in df.columns: df["description"] = ""
    elif ledger_type == "bank":
        if "bank_ref" not in df.columns: df["bank_ref"] = [f"BANK-{i+1:05d}" for i in range(len(df))]
        if "amount" not in df.columns: df["amount"] = 0.0
        if "date" not in df.columns: df["date"] = datetime.now().strftime("%Y-%m-%d")
        if "currency" not in df.columns: df["currency"] = "INR"
        if "description" not in df.columns: df["description"] = ""
        if "type" not in df.columns: df["type"] = "CREDIT"
    elif ledger_type == "gateway":
        if "gw_ref" not in df.columns: df["gw_ref"] = [f"GW-{i+1:05d}" for i in range(len(df))]
        if "amount" not in df.columns: df["amount"] = 0.0
        if "date" not in df.columns: df["date"] = datetime.now().strftime("%Y-%m-%d")
        if "currency" not in df.columns: df["currency"] = "INR"
        if "fee" not in df.columns: df["fee"] = 0.0
        if "erp_ref_hint" not in df.columns: df["erp_ref_hint"] = ""

    # Ensure clean types
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["date"] = df["date"].astype(str).str.strip()
    return df


def parse_date(date_str):
    """Safely parse a date string into a datetime object."""
    if not date_str or pd.isna(date_str):
        return None
    s = str(date_str).strip().split(" ")[0]
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def amount_similarity(a1, a2, tolerance_pct=0.05):
    """
    Compute how similar two amounts are.
    Returns a score between 0 (identical) and 1 (very different).
    """
    if a1 == 0 and a2 == 0:
        return 0.0
    if a1 <= 0 or a2 <= 0:
        return 1.0

    pct_diff = abs(a1 - a2) / max(abs(a1), abs(a2))
    if pct_diff > tolerance_pct:
        return 1.0
    return pct_diff


def date_similarity(d1, d2, max_lag_days=7):
    """
    Compute how similar two dates are.
    Returns 0 (same day) to 1 (more than max_lag_days apart).
    """
    if d1 is None or d2 is None:
        return 1.0

    lag = abs((d2 - d1).days)
    if lag > max_lag_days:
        return 1.0
    return lag / max_lag_days


def reference_similarity(ref1, ref2):
    """
    Compute string similarity between two reference IDs.
    Returns 0 (identical) to 1 (completely different).
    """
    if not ref1 or not ref2 or pd.isna(ref1) or pd.isna(ref2):
        return 1.0

    score = fuzz.partial_ratio(str(ref1).upper(), str(ref2).upper())
    return 1.0 - (score / 100.0)


def compute_pair_cost(
    erp_amount, bank_amount,
    erp_date, bank_date,
    erp_ref, bank_ref,
    w_amount=0.5, w_date=0.3, w_ref=0.2
):
    cost_amount = amount_similarity(erp_amount, bank_amount)
    cost_date   = date_similarity(erp_date, bank_date)
    cost_ref    = reference_similarity(erp_ref, bank_ref)

    total_cost = (w_amount * cost_amount +
                  w_date   * cost_date   +
                  w_ref    * cost_ref)

    return round(total_cost, 4)


def generate_candidate_pairs(
    erp_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    gw_df: pd.DataFrame = None,
    max_cost: float = 0.60
) -> pd.DataFrame:
    """
    Shortlists plausible matching pairs between ERP, Bank, and Gateway ledgers.
    """
    erp_df = normalize_ledger_columns(erp_df, "erp")
    bank_df = normalize_ledger_columns(bank_df, "bank")
    if gw_df is not None and not gw_df.empty:
        gw_df = normalize_ledger_columns(gw_df, "gateway")

    candidates = []

    erp_dates  = {row["invoice_id"]: parse_date(row["date"]) for _, row in erp_df.iterrows()}
    bank_dates = {row["bank_ref"]: parse_date(row["date"]) for _, row in bank_df.iterrows()}
    gw_dates   = {row["gw_ref"]: parse_date(row["date"]) for _, row in gw_df.iterrows()} if gw_df is not None and not gw_df.empty else {}

    # 1. ERP <-> Bank Candidates
    for _, erp_row in erp_df.iterrows():
        erp_id     = str(erp_row["invoice_id"])
        erp_amount = float(erp_row["amount"])
        erp_date   = erp_dates[erp_id]
        erp_curr   = str(erp_row.get("currency", "INR")).upper()

        for _, bank_row in bank_df.iterrows():
            bank_ref    = str(bank_row["bank_ref"])
            bank_amount = float(bank_row["amount"])
            bank_date   = bank_dates[bank_ref]
            bank_curr   = str(bank_row.get("currency", "INR")).upper()

            # Pre-filter: amount delta > 25% skip
            if erp_amount > 0:
                quick_pct = abs(erp_amount - bank_amount) / erp_amount
                if quick_pct > 0.25:
                    continue

            cost = compute_pair_cost(
                erp_amount, bank_amount,
                erp_date, bank_date,
                erp_id, bank_ref
            )

            # Find matching gateway record if available
            gw_ref_match = None
            if gw_df is not None and not gw_df.empty:
                # Find gateway record with hint or matching amount
                gw_candidates = gw_df[
                    (gw_df["erp_ref_hint"].astype(str).str.upper() == erp_id.upper()) |
                    (abs(gw_df["amount"] + gw_df["fee"] - erp_amount) <= 5.0)
                ]
                if not gw_candidates.empty:
                    gw_ref_match = str(gw_candidates.iloc[0]["gw_ref"])

            if cost <= max_cost:
                candidates.append({
                    "erp_invoice_id": erp_id,
                    "bank_ref":       bank_ref,
                    "gw_ref":         gw_ref_match if gw_ref_match else "NONE",
                    "erp_amount":     erp_amount,
                    "bank_amount":    bank_amount,
                    "erp_date":       str(erp_row["date"]),
                    "bank_date":      str(bank_row["date"]),
                    "erp_currency":   erp_curr,
                    "bank_currency":  bank_curr,
                    "cost":           cost,
                    "cost_amount":    amount_similarity(erp_amount, bank_amount),
                    "cost_date":      date_similarity(erp_date, bank_date),
                    "cost_ref":       reference_similarity(erp_id, bank_ref),
                })

    candidates_df = pd.DataFrame(candidates)
    return candidates_df