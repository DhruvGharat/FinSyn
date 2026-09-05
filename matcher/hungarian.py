import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
from matcher.split_merge import find_split_matches, find_merged_matches


def build_cost_matrix(
    erp_df:        pd.DataFrame,
    bank_df:       pd.DataFrame,
    candidates_df: pd.DataFrame
) -> tuple:
    """
    Builds a cost matrix from candidate pairs.
    Rows    = ERP invoices
    Columns = Bank rows
    Value   = cost score (0 = perfect match, 1 = no match)
    """
    erp_ids   = erp_df["invoice_id"].astype(str).tolist()
    bank_refs = bank_df["bank_ref"].astype(str).tolist()

    n_erp  = len(erp_ids)
    n_bank = len(bank_refs)

    cost_matrix = np.ones((n_erp, n_bank), dtype=float)

    erp_idx  = {erp_id:   i for i, erp_id  in enumerate(erp_ids)}
    bank_idx = {bank_ref: j for j, bank_ref in enumerate(bank_refs)}

    if not candidates_df.empty:
        for _, row in candidates_df.iterrows():
            i = erp_idx.get(str(row["erp_invoice_id"]))
            j = bank_idx.get(str(row["bank_ref"]))
            if i is not None and j is not None:
                cost_matrix[i, j] = float(row["cost"])

    return cost_matrix, erp_ids, bank_refs


def run_hungarian(
    erp_df:        pd.DataFrame,
    bank_df:       pd.DataFrame,
    candidates_df: pd.DataFrame,
    max_accept_cost: float = 0.55
) -> pd.DataFrame:
    """
    Runs the Hungarian algorithm globally across ERP and Bank ledgers.
    Integrates Split and Merged payment logic.
    """
    # 1. First check for Split and Merged payments
    split_matches = find_split_matches(erp_df, bank_df)
    merged_matches = find_merged_matches(erp_df, bank_df)

    matched_split_erp = {s["erp_invoice_id"] for s in split_matches}
    matched_split_bank = {ref for s in split_matches for ref in s["bank_refs"]}

    matched_merged_erp = {eid for m in merged_matches for eid in m["erp_invoice_ids"]}
    matched_merged_bank = {m["bank_ref"] for m in merged_matches}

    # Filter out ERP/Bank rows already solved by split/merge logic
    clean_erp = erp_df[~erp_df["invoice_id"].astype(str).isin(matched_split_erp | matched_merged_erp)]
    clean_bank = bank_df[~bank_df["bank_ref"].astype(str).isin(matched_split_bank | matched_merged_bank)]

    cost_matrix, erp_ids, bank_refs = build_cost_matrix(
        clean_erp, clean_bank, candidates_df
    )

    results = []

    # Include Split & Merged matches into results
    for s in split_matches:
        results.append({
            "erp_invoice_id": s["erp_invoice_id"],
            "bank_ref": ",".join(s["bank_refs"]),
            "cost": s["cost"],
            "decision": "MATCHED_SPLIT",
            "match_type": "SPLIT_1_TO_MANY"
        })

    for m in merged_matches:
        for eid in m["erp_invoice_ids"]:
            results.append({
                "erp_invoice_id": eid,
                "bank_ref": m["bank_ref"],
                "cost": m["cost"],
                "decision": "MATCHED_MERGED",
                "match_type": "MERGED_MANY_TO_1"
            })

    if cost_matrix.size > 0:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row_i, col_j in zip(row_indices, col_indices):
            erp_id   = erp_ids[row_i]
            bank_ref = bank_refs[col_j]
            cost     = cost_matrix[row_i, col_j]

            if cost >= max_accept_cost:
                decision = "NO_MATCH"
                assigned_bank = None
            else:
                decision = "MATCHED"
                assigned_bank = bank_ref

            results.append({
                "erp_invoice_id": erp_id,
                "bank_ref":       assigned_bank,
                "cost":           round(float(cost), 4),
                "decision":       decision,
                "match_type":     "ONE_TO_ONE" if decision == "MATCHED" else "UNMATCHED"
            })

    # Unassigned ERP rows
    assigned_erp = {r["erp_invoice_id"] for r in results}
    all_erp_ids = erp_df["invoice_id"].astype(str).tolist()
    for erp_id in all_erp_ids:
        if erp_id not in assigned_erp:
            results.append({
                "erp_invoice_id": erp_id,
                "bank_ref":       None,
                "cost":           1.0,
                "decision":       "NO_MATCH",
                "match_type":     "UNMATCHED"
            })

    return pd.DataFrame(results)


def evaluate_against_ground_truth(
    assignments_df: pd.DataFrame,
    ground_truth:   dict
) -> dict:
    """
    Evaluates assigned predictions against ground truth labels.
    """
    true_positives  = 0
    false_positives = 0
    false_negatives = 0
    true_negatives  = 0

    for _, row in assignments_df.iterrows():
        erp_id    = str(row["erp_invoice_id"])
        predicted = str(row["bank_ref"]) if row["bank_ref"] else None
        decision  = row["decision"]
        gt_info   = ground_truth.get(erp_id)

        if isinstance(gt_info, dict):
            true_bank_refs = gt_info.get("bank_refs", [])
        elif isinstance(gt_info, list):
            true_bank_refs = gt_info
        elif isinstance(gt_info, str):
            true_bank_refs = [gt_info]
        else:
            true_bank_refs = []

        if decision in ["MATCHED", "MATCHED_SPLIT", "MATCHED_MERGED"]:
            if not true_bank_refs:
                false_positives += 1
            else:
                # Check if predicted bank ref overlaps true bank refs
                pred_list = predicted.split(",") if predicted else []
                if any(p in true_bank_refs for p in pred_list):
                    true_positives += 1
                else:
                    false_positives += 1
        else: # NO_MATCH
            if true_bank_refs:
                false_negatives += 1
            else:
                true_negatives += 1

    precision = (true_positives / (true_positives + false_positives)
                 if (true_positives + false_positives) > 0 else 0.0)
    recall    = (true_positives / (true_positives + false_negatives)
                 if (true_positives + false_negatives) > 0 else 0.0)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives":  true_negatives,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
    }