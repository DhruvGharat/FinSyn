import pandas as pd
import numpy as np
from scipy.optimize import linprog
from datetime import datetime
from matcher.engine import parse_date


def find_split_matches(erp_df: pd.DataFrame, bank_df: pd.DataFrame, max_date_lag: int = 5, amount_tolerance: float = 0.01) -> list:
    """
    Identifies 1 ERP -> Many Bank split matches (e.g. 1 ERP invoice split into 2 bank deposits).
    Uses subset-sum matching combined with date proximity.
    """
    split_matches = []
    
    # Pre-parse dates
    bank_df = bank_df.copy()
    bank_df["_parsed_date"] = bank_df["date"].apply(parse_date)
    
    for _, erp_row in erp_df.iterrows():
        erp_id = erp_row["invoice_id"]
        erp_amt = float(erp_row["amount"])
        erp_date = parse_date(erp_row["date"])
        if not erp_date or erp_amt <= 0:
            continue

        # Candidate bank rows within date window
        valid_bank = []
        for idx, bank_row in bank_df.iterrows():
            b_date = bank_row["_parsed_date"]
            if b_date and 0 <= (b_date - erp_date).days <= max_date_lag:
                valid_bank.append((bank_row["bank_ref"], float(bank_row["amount"]), b_date))
        
        # Check pairs of bank rows that sum up to erp_amt
        n = len(valid_bank)
        found_split = False
        for i in range(n):
            if found_split:
                break
            for j in range(i + 1, n):
                ref1, amt1, d1 = valid_bank[i]
                ref2, amt2, d2 = valid_bank[j]
                if abs((amt1 + amt2) - erp_amt) / max(erp_amt, 1.0) <= amount_tolerance:
                    split_matches.append({
                        "erp_invoice_id": erp_id,
                        "bank_refs": [ref1, ref2],
                        "erp_amount": erp_amt,
                        "bank_amounts": [amt1, amt2],
                        "type": "SPLIT_1_TO_MANY",
                        "cost": 0.02
                    })
                    found_split = True
                    break

    return split_matches


def find_merged_matches(erp_df: pd.DataFrame, bank_df: pd.DataFrame, max_date_lag: int = 5, amount_tolerance: float = 0.01) -> list:
    """
    Identifies Many ERP -> 1 Bank merged/batch matches (e.g. 2 ERP invoices paid in 1 bank deposit).
    """
    merged_matches = []
    
    erp_df = erp_df.copy()
    erp_df["_parsed_date"] = erp_df["date"].apply(parse_date)

    for _, bank_row in bank_df.iterrows():
        bank_ref = bank_row["bank_ref"]
        bank_amt = float(bank_row["amount"])
        bank_date = parse_date(bank_row["date"])
        if not bank_date or bank_amt <= 0:
            continue

        # Candidate ERP rows within date window
        valid_erp = []
        for idx, erp_row in erp_df.iterrows():
            e_date = erp_row["_parsed_date"]
            if e_date and 0 <= (bank_date - e_date).days <= max_date_lag:
                valid_erp.append((erp_row["invoice_id"], float(erp_row["amount"]), e_date))
        
        n = len(valid_erp)
        found_merged = False
        for i in range(n):
            if found_merged:
                break
            for j in range(i + 1, n):
                id1, amt1, d1 = valid_erp[i]
                id2, amt2, d2 = valid_erp[j]
                if abs((amt1 + amt2) - bank_amt) / max(bank_amt, 1.0) <= amount_tolerance:
                    merged_matches.append({
                        "erp_invoice_ids": [id1, id2],
                        "bank_ref": bank_ref,
                        "erp_amounts": [amt1, amt2],
                        "bank_amount": bank_amt,
                        "type": "MERGED_MANY_TO_1",
                        "cost": 0.02
                    })
                    found_merged = True
                    break

    return merged_matches


def solve_transportation_split(erp_df: pd.DataFrame, bank_df: pd.DataFrame, cost_matrix: np.ndarray) -> dict:
    """
    Transportation LP solver formulation using scipy.optimize.linprog for fractional assignment.
    """
    n_erp, n_bank = cost_matrix.shape
    if n_erp == 0 or n_bank == 0:
        return {}

    c = cost_matrix.flatten()
    
    A_eq = []
    b_eq = []
    
    for i in range(n_erp):
        row = np.zeros((n_erp, n_bank))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(1)
        
    for j in range(n_bank):
        col = np.zeros((n_erp, n_bank))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(1)
        
    bounds = [(0, 1) for _ in range(n_erp * n_bank)]
    
    try:
        res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
        if res.success:
            X = res.x.reshape((n_erp, n_bank))
            return {"status": "OPTIMAL", "assignment_matrix": X}
    except Exception:
        pass
        
    return {"status": "FAILED", "assignment_matrix": None}
