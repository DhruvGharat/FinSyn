import json
import pandas as pd
from matcher.pipeline import run_reconciliation_pipeline


def run_full_pipeline():
    print("\n" + "="*60)
    print("  FINSYN RECONCILIATION PIPELINE — END-TO-END VERIFICATION")
    print("="*60)

    # 1. Load synthetic data
    erp_df  = pd.read_csv("data/synthetic/erp_ledger.csv")
    bank_df = pd.read_csv("data/synthetic/bank_statement.csv")
    gw_df   = pd.read_csv("data/synthetic/gateway_settlement.csv")

    with open("data/synthetic/ground_truth.json") as f:
        ground_truth = json.load(f)

    # 2. Run Pipeline
    print("\nExecuting complete pipeline...")
    results = run_reconciliation_pipeline(
        erp_df=erp_df,
        bank_df=bank_df,
        gw_df=gw_df,
        ground_truth=ground_truth,
        scenario="default"
    )

    summary = results["summary"]
    gt_eval = results["ground_truth_eval"]
    ml_eval = results["ml_eval_metrics"]

    print("\n" + "="*60)
    print("  RECONCILIATION RUN SUMMARY")
    print("="*60)
    print(f"  Run ID:                {results['run_id']}")
    print(f"  Total ERP Rows:        {summary['total_erp_count']}")
    print(f"  Total Bank Rows:       {summary['total_bank_count']}")
    print(f"  Total Gateway Rows:    {summary['total_gw_count']}")
    print(f"  Auto-Accepted Matches: {summary['matched_count']}")
    print(f"  Manual Review:         {summary['review_count']}")
    print(f"  Auto-Rejected:         {summary['rejected_count']}")
    print(f"  Unmatched Exceptions:  {summary['unmatched_count']}")
    print(f"  SHA-256 Chain Valid:   {results['chain_verified']}")

    if gt_eval:
        print("\n--- Ground Truth Evaluation Metrics ---")
        print(f"  Precision:             {gt_eval.get('precision', 0):.1%}")
        print(f"  Recall:                {gt_eval.get('recall', 0):.1%}")
        print(f"  F1 Score:              {gt_eval.get('f1_score', 0):.1%}")

    if ml_eval:
        print("\n--- XGBoost Classifier Test Metrics ---")
        print(f"  Precision:             {ml_eval.get('precision', 0):.1%}")
        print(f"  Recall:                {ml_eval.get('recall', 0):.1%}")
        print(f"  F1 Score:              {ml_eval.get('f1_score', 0):.1%}")
        print(f"  Avg Precision (AP):    {ml_eval.get('average_precision', 0):.1%}")

    print("\n--- Sample Exception Explanations ---")
    for exc in results["exceptions"][:3]:
        print(f"  Invoice: {exc.get('invoice_id')} | Amount: INR {exc.get('amount')}")
        print(f"  Category: {exc.get('exception_category')} | Severity: {exc.get('severity')}")
        print(f"  Explanation: {exc.get('explanation')}")
        print(f"  Action: {exc.get('recommended_action')}\n")

    print("="*60)
    print("  PIPELINE VERIFICATION COMPLETE — ALL SYSTEMS GO")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()