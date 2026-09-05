import os
import io
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd

from matcher.pipeline import run_reconciliation_pipeline
from generator.synthetic import generate_transaction_set

app = FastAPI(
    title="Razorpay AI Finance Controller API",
    description="Automated financial reconciliation system powered by Hungarian Algorithm, XGBoost, and LLM Audit Agents.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "service": "Razorpay AI Finance Controller",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }



@app.post("/reconcile")
async def reconcile_ledgers(
    erp_file: UploadFile = File(..., description="ERP Ledger CSV"),
    bank_file: UploadFile = File(..., description="Bank Statement CSV"),
    gateway_file: UploadFile = File(None, description="Gateway Settlement CSV (Optional)"),
    scenario: str = Form("default")
):
    # File type & extension validation
    if not erp_file.filename.endswith(".csv") or not bank_file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload valid .csv files for ERP and Bank ledgers."
        )

    if gateway_file and gateway_file.filename and not gateway_file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format for Gateway settlement. Please upload a valid .csv file."
        )

    try:
        erp_bytes = await erp_file.read()
        bank_bytes = await bank_file.read()

        if len(erp_bytes) == 0 or len(bank_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV files cannot be empty."
            )

        erp_df = pd.read_csv(io.BytesIO(erp_bytes))
        bank_df = pd.read_csv(io.BytesIO(bank_bytes))

        gw_df = None
        if gateway_file and gateway_file.filename:
            gw_bytes = await gateway_file.read()
            if len(gw_bytes) > 0:
                gw_df = pd.read_csv(io.BytesIO(gw_bytes))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV files: {str(e)}"
        )

    try:
        results = run_reconciliation_pipeline(
            erp_df=erp_df,
            bank_df=bank_df,
            gw_df=gw_df,
            scenario=scenario
        )

        return JSONResponse(content={
            "run_id": results["run_id"],
            "status": "SUCCESS",
            "summary": results["summary"],
            "decisions_count": len(results["decisions"]),
            "exceptions_count": len(results["exceptions"]),
            "decisions": results["decisions"][:50],  # Sample response for payload size
            "exceptions": results["exceptions"][:50],
            "chain_verified": results["chain_verified"],
            "ml_eval_metrics": results["ml_eval_metrics"]
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconciliation processing error: {str(e)}"
        )


@app.post("/demo/generate")
def generate_demo(n_transactions: int = 500, scenario: str = "default"):
    try:
        erp_df, bank_df, gw_df, ground_truth = generate_transaction_set(
            n_transactions=n_transactions, seed=42, scenario=scenario
        )
        return {
            "status": "SUCCESS",
            "scenario": scenario,
            "counts": {
                "erp": len(erp_df),
                "bank": len(bank_df),
                "gateway": len(gw_df)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo generation failed: {str(e)}"
        )
