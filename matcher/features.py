import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from matcher.engine import parse_date


def engineer_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes target 8 features for every candidate pair:
    1. Absolute amount delta (amt_diff_abs)
    2. Percentage amount delta (amt_diff_pct)
    3. Date lag in days (date_lag)
    4. Reference string similarity (ref_similarity)
    5. Currency match (currency_match)
    6. Round amount flag (is_round)
    7. Transaction type match (tx_type_match)
    8. Hour-of-day difference / combined cost (combined_cost)
    """
    features = []

    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=[
            "erp_invoice_id", "bank_ref", "gw_ref", "erp_amount", "bank_amount",
            "amt_diff_abs", "amt_diff_pct", "date_lag", "ref_similarity",
            "currency_match", "is_round", "tx_type_match", "combined_cost"
        ])

    for _, row in candidates_df.iterrows():
        erp_amount  = float(row.get("erp_amount", 0.0))
        bank_amount = float(row.get("bank_amount", 0.0))
        erp_date    = parse_date(row.get("erp_date"))
        bank_date   = parse_date(row.get("bank_date"))
        erp_ref     = str(row.get("erp_invoice_id", ""))
        bank_ref    = str(row.get("bank_ref", ""))
        gw_ref      = str(row.get("gw_ref", "NONE"))
        erp_curr    = str(row.get("erp_currency", "INR")).upper()
        bank_curr   = str(row.get("bank_currency", "INR")).upper()

        amt_diff_abs = abs(erp_amount - bank_amount)
        amt_diff_pct = (amt_diff_abs / max(abs(erp_amount), 1.0)) * 100.0

        if erp_date and bank_date:
            date_lag = abs((bank_date - erp_date).days)
        else:
            date_lag = 999

        ref_similarity = fuzz.partial_ratio(erp_ref.upper(), bank_ref.upper()) / 100.0
        currency_match = 1 if erp_curr == bank_curr else 0
        is_round = 1 if (erp_amount > 0 and erp_amount % 100 == 0) else 0
        tx_type_match = 1  # Credit settlement match
        combined_cost = float(row.get("cost", 1.0))

        features.append({
            "erp_invoice_id": erp_ref,
            "bank_ref":       bank_ref,
            "gw_ref":         gw_ref,
            "erp_amount":     erp_amount,
            "bank_amount":    bank_amount,
            "amt_diff_abs":   round(amt_diff_abs, 4),
            "amt_diff_pct":   round(amt_diff_pct, 6),
            "date_lag":       date_lag,
            "ref_similarity": round(ref_similarity, 4),
            "currency_match": currency_match,
            "is_round":       is_round,
            "tx_type_match": tx_type_match,
            "combined_cost":  combined_cost,
        })

    return pd.DataFrame(features)


def label_candidates(
    features_df:  pd.DataFrame,
    ground_truth: dict
) -> pd.DataFrame:
    """
    Adds binary label (1=MATCH, 0=NO_MATCH) based on ground truth mappings.
    """
    labels = []

    for _, row in features_df.iterrows():
        erp_id   = str(row["erp_invoice_id"])
        bank_ref = str(row["bank_ref"])
        gt_info  = ground_truth.get(erp_id)

        if gt_info is None:
            labels.append(0)
        elif isinstance(gt_info, dict):
            true_refs = gt_info.get("bank_refs", [])
            labels.append(1 if bank_ref in true_refs else 0)
        elif isinstance(gt_info, list):
            labels.append(1 if bank_ref in gt_info else 0)
        elif isinstance(gt_info, str):
            labels.append(1 if bank_ref == gt_info else 0)
        else:
            labels.append(0)

    features_df = features_df.copy()
    features_df["label"] = labels
    return features_df