import pandas as pd
import numpy as np
import random
import string
import json
from datetime import datetime, timedelta
from pathlib import Path


def random_id(prefix="", length=8):
    """Generate a random alphanumeric ID like GW-A1B2C3D4"""
    chars = string.ascii_uppercase + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def random_date(start="2026-01-01", end="2026-03-31"):
    """Pick a random date between start and end"""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    return start_dt + timedelta(days=random.randint(0, delta))


def generate_transaction_set(
    n_transactions=500,
    seed=42,
    scenario="default",
    output_dir="data/synthetic"
):
    """
    Generates three financial ledgers (ERP, Bank, Gateway) with ground truth.
    Supports specific demo scenarios:
    - default: Realistic mix of all noise types
    - clean: 95%+ exact matches
    - timing_rounding: High date lag and amount rounding
    - ambiguous: Similar amounts and missing reference IDs
    - split_merge: High split/merge transaction rate
    - missing: High rate of missing bank/gateway records
    """
    random.seed(seed)
    np.random.seed(seed)

    # Configure noise parameters based on scenario
    if scenario == "clean":
        fx_noise_pct    = 0.01
        timing_lag_pct  = 0.05
        id_mismatch_pct = 0.10
        split_pct       = 0.00
        missing_pct     = 0.01
    elif scenario == "timing_rounding":
        fx_noise_pct    = 0.35
        timing_lag_pct  = 0.60
        id_mismatch_pct = 0.50
        split_pct       = 0.02
        missing_pct     = 0.02
    elif scenario == "ambiguous":
        fx_noise_pct    = 0.20
        timing_lag_pct  = 0.20
        id_mismatch_pct = 0.90
        split_pct       = 0.05
        missing_pct     = 0.05
    elif scenario == "split_merge":
        fx_noise_pct    = 0.10
        timing_lag_pct  = 0.20
        id_mismatch_pct = 0.60
        split_pct       = 0.20
        missing_pct     = 0.05
    elif scenario == "missing":
        fx_noise_pct    = 0.10
        timing_lag_pct  = 0.20
        id_mismatch_pct = 0.50
        split_pct       = 0.02
        missing_pct     = 0.25
    else:  # default realistic mix
        fx_noise_pct    = 0.10
        timing_lag_pct  = 0.30
        id_mismatch_pct = 0.60
        split_pct       = 0.05
        missing_pct     = 0.05

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    erp_rows  = []
    bank_rows = []
    gw_rows   = []
    ground_truth = {}

    for i in range(n_transactions):
        # 30% of transactions use round amounts (higher risk of ambiguity)
        if scenario == "ambiguous" or random.random() < 0.35:
            true_amount = float(random.choice([
                500, 1000, 1500, 2000, 2500, 3000, 5000,
                7500, 10000, 15000, 20000, 25000, 50000
            ]))
        else:
            true_amount = round(random.uniform(500, 100000), 2)

        true_date   = random_date()
        erp_invoice = f"INV-{i+1:05d}"
        customer    = f"CUST-{random.randint(1000, 9999)}"
        currency    = "INR"

        erp_amount = true_amount
        if random.random() < (fx_noise_pct * 0.5):
            erp_amount = float(round(true_amount))

        erp_rows.append({
            "invoice_id":  erp_invoice,
            "date":        true_date.strftime("%Y-%m-%d"),
            "amount":      erp_amount,
            "currency":    currency,
            "customer_id": customer,
            "description": f"Invoice payment {erp_invoice}",
            "status":      "PAID"
        })

        introduce_split   = (random.random() < split_pct)
        introduce_missing = (not introduce_split and random.random() < missing_pct)

        if random.random() < timing_lag_pct:
            bank_date = true_date + timedelta(days=random.choice([1, 2, 2, 3, 4]))
        else:
            bank_date = true_date

        if random.random() < fx_noise_pct:
            noise = random.uniform(-2.5, 2.5)
            bank_amount = round(max(1.0, true_amount + noise), 2)
        else:
            bank_amount = true_amount

        if random.random() < id_mismatch_pct:
            bank_ref = f"RZRPY-{random_id(length=8)}"
        else:
            bank_ref = erp_invoice

        gw_ref = f"GW-{random_id(length=10)}"
        gw_date = true_date + timedelta(days=random.choice([1, 2]))
        gw_fee = round(true_amount * random.uniform(0.015, 0.025), 2)
        gw_amount = round(true_amount - gw_fee, 2)

        if introduce_split:
            half1 = round(bank_amount * 0.4, 2)
            half2 = round(bank_amount - half1, 2)
            bank_ref_1 = f"RZRPY-{random_id(length=8)}"
            bank_ref_2 = f"RZRPY-{random_id(length=8)}"
            bank_rows.append({
                "bank_ref":    bank_ref_1,
                "date":        bank_date.strftime("%Y-%m-%d"),
                "amount":      half1,
                "currency":    currency,
                "description": f"RAZORPAY SETTLEMENT SPLIT 1 {bank_ref_1}",
                "type":        "CREDIT"
            })
            bank_rows.append({
                "bank_ref":    bank_ref_2,
                "date":        (bank_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                "amount":      half2,
                "currency":    currency,
                "description": f"RAZORPAY SETTLEMENT SPLIT 2 {bank_ref_2}",
                "type":        "CREDIT"
            })
            ground_truth[erp_invoice] = {
                "bank_refs": [bank_ref_1, bank_ref_2],
                "gw_ref": gw_ref,
                "type": "SPLIT"
            }
        elif introduce_missing:
            ground_truth[erp_invoice] = {
                "bank_refs": [],
                "gw_ref": None,
                "type": "MISSING"
            }
        else:
            bank_rows.append({
                "bank_ref":    bank_ref,
                "date":        bank_date.strftime("%Y-%m-%d"),
                "amount":      bank_amount,
                "currency":    currency,
                "description": f"RAZORPAY SETTLEMENT {bank_ref}",
                "type":        "CREDIT"
            })
            ground_truth[erp_invoice] = {
                "bank_refs": [bank_ref],
                "gw_ref": gw_ref,
                "type": "EXACT" if bank_ref == erp_invoice else "NEAR_MATCH"
            }

        if not introduce_missing:
            gw_rows.append({
                "gw_ref":       gw_ref,
                "date":         gw_date.strftime("%Y-%m-%d"),
                "amount":       gw_amount,
                "fee":          gw_fee,
                "currency":     currency,
                "erp_ref_hint": erp_invoice if random.random() > 0.4 else "",
                "status":       "SETTLED"
            })

    erp_df  = pd.DataFrame(erp_rows)
    bank_df = pd.DataFrame(bank_rows)
    gw_df   = pd.DataFrame(gw_rows)

    erp_path  = f"{output_dir}/erp_ledger.csv"
    bank_path = f"{output_dir}/bank_statement.csv"
    gw_path   = f"{output_dir}/gateway_settlement.csv"
    gt_path   = f"{output_dir}/ground_truth.json"

    erp_df.to_csv(erp_path, index=False)
    bank_df.to_csv(bank_path, index=False)
    gw_df.to_csv(gw_path, index=False)

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return erp_df, bank_df, gw_df, ground_truth


def generate_dataset(n_transactions=500, output_dir="data/synthetic"):
    return generate_transaction_set(n_transactions=n_transactions, seed=42, scenario="default", output_dir=output_dir)


if __name__ == "__main__":
    generate_transaction_set(n_transactions=500, seed=42, scenario="default")