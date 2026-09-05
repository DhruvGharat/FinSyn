import os
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page layout & custom CSS
st.set_page_config(
    page_title="Razorpay AI Finance Controller",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for financial dashboard
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stMetric label { color: #9CA3AF !important; font-size: 0.9rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #F3F4F6 !important; font-weight: 700 !important; }
    .badge-success { background-color: #065F46; color: #34D399; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-warning { background-color: #92400E; color: #FBBF24; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-danger { background-color: #991B1B; color: #FCA5A5; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator.synthetic import generate_transaction_set
from matcher.pipeline import run_reconciliation_pipeline

st.title("⚡ Razorpay AI Finance Controller")
st.caption("Autonomous Financial Reconciliation Engine — Deterministic Optimization + XGBoost + Skeptical LLM Auditor")

# Initialize session state
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "demo_generated" not in st.session_state:
    st.session_state["demo_generated"] = False

# ── SIDEBAR CONTROL PANEL ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls & Uploads")

    st.subheader("1. Demo Generator")
    demo_scenario = st.selectbox(
        "Select Demo Scenario",
        ["default", "clean", "timing_rounding", "ambiguous", "split_merge", "missing"],
        help="Generates synthetic ERP, Bank, and Gateway ledgers with known ground truth."
    )
    n_txns = st.slider("Transaction Volume", 50, 1000, 300, step=50)

    if st.button("🔄 Generate Demo Dataset", use_container_width=True):
        with st.spinner("Generating reproducible synthetic ledgers..."):
            generate_transaction_set(n_transactions=n_txns, seed=42, scenario=demo_scenario)
            st.session_state["demo_generated"] = True
            st.session_state["pipeline_results"] = None
            st.success(f"Generated {n_txns} transactions for scenario '{demo_scenario}'!")

    st.divider()
    st.subheader("2. CSV Upload (Custom)")
    uploaded_erp = st.file_uploader("ERP Ledger CSV", type=["csv"])
    uploaded_bank = st.file_uploader("Bank Statement CSV", type=["csv"])
    uploaded_gw = st.file_uploader("Gateway Settlement CSV (Optional)", type=["csv"])

    st.divider()
    run_btn = st.button("🚀 Run Reconciliation", type="primary", use_container_width=True)

# ── PIPELINE EXECUTION ────────────────────────────────────────────
if run_btn:
    with st.spinner("Executing 3-Way Reconciliation Pipeline..."):
        ground_truth = None

        if uploaded_erp and uploaded_bank:
            erp_df = pd.read_csv(uploaded_erp)
            bank_df = pd.read_csv(uploaded_bank)
            gw_df = pd.read_csv(uploaded_gw) if uploaded_gw else None
            scenario_name = "Uploaded CSVs"
        elif os.path.exists("data/synthetic/erp_ledger.csv"):
            erp_df = pd.read_csv("data/synthetic/erp_ledger.csv")
            bank_df = pd.read_csv("data/synthetic/bank_statement.csv")
            gw_df = pd.read_csv("data/synthetic/gateway_settlement.csv") if os.path.exists("data/synthetic/gateway_settlement.csv") else None
            if os.path.exists("data/synthetic/ground_truth.json"):
                with open("data/synthetic/ground_truth.json") as f:
                    ground_truth = json.load(f)
            scenario_name = demo_scenario
        else:
            erp_df, bank_df, gw_df, ground_truth = generate_transaction_set(n_transactions=300, seed=42, scenario="default")
            scenario_name = "default"

        results = run_reconciliation_pipeline(
            erp_df=erp_df,
            bank_df=bank_df,
            gw_df=gw_df,
            ground_truth=ground_truth,
            scenario=scenario_name
        )
        st.session_state["pipeline_results"] = results
        st.success("Reconciliation completed successfully!")

results = st.session_state.get("pipeline_results")

if results:
    summary = results["summary"]
    decisions = results["decisions"]
    exceptions = results["exceptions"]
    gt_eval = results.get("ground_truth_eval", {})
    ml_eval = results.get("ml_eval_metrics", {})
    dec_df = pd.DataFrame(decisions)
    exc_df = pd.DataFrame(exceptions)

    # Calculate match rate
    total_tx = summary["total_erp_count"]
    matched_tx = summary["matched_count"]
    match_rate = (matched_tx / total_tx * 100.0) if total_tx > 0 else 0.0

    # ── TOP KPI CARDS ──────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total Transactions", total_tx)
    col2.metric("Matched", matched_tx)
    col3.metric("Exceptions", summary["unmatched_count"])
    col4.metric("Auto Accepted", summary["matched_count"])
    col5.metric("Manual Review", summary["review_count"])
    col6.metric("Auto Rejected", summary["rejected_count"])
    col7.metric("Match Rate", f"{match_rate:.1f}%")

    st.markdown("---")

    # ── DASHBOARD NAVIGATION TABS ───────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📊 Overview",
        "📋 Match Report",
        "⚠️ Exceptions",
        "🔍 Adversarial Audit",
        "📈 ML Evaluation",
        "🔒 Audit Trail",
        "📥 Downloads"
    ])

    # ── TAB 1: OVERVIEW ────────────────────────────────────────────
    with t1:
        st.subheader("Reconciliation Overview & Confidence Distribution")
        c1, c2 = st.columns([1, 1])

        with c1:
            if not dec_df.empty and "confidence_zone" in dec_df.columns:
                zone_counts = dec_df["confidence_zone"].value_counts().reset_index()
                zone_counts.columns = ["Zone", "Count"]
                fig_pie = px.pie(
                    zone_counts, values="Count", names="Zone",
                    title="Confidence Zone Breakdown",
                    color="Zone",
                    color_discrete_map={"HIGH": "#10B981", "MEDIUM": "#F59E0B", "LOW": "#EF4444"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if not dec_df.empty and "confidence" in dec_df.columns:
                fig_hist = px.histogram(
                    dec_df, x="confidence", nbins=20,
                    title="XGBoost Match Confidence Score Distribution",
                    color_discrete_sequence=["#6366F1"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 2: MATCH REPORT ────────────────────────────────────────
    with t2:
        st.subheader("Match Decision Ledger")
        if not dec_df.empty:
            cols_to_show = ["erp_invoice_id", "bank_ref", "gw_ref", "erp_amount", "bank_amount", "confidence", "confidence_zone", "final_decision", "match_type"]
            present_cols = [c for c in cols_to_show if c in dec_df.columns]
            st.dataframe(dec_df[present_cols], use_container_width=True)
        else:
            st.info("No candidate match decisions generated.")

    # ── TAB 3: EXCEPTIONS ──────────────────────────────────────────
    with t3:
        st.subheader("Unmatched Transactions & LLM Explanations")
        if not exc_df.empty:
            for idx, row in exc_df.iterrows():
                with st.expander(f"Invoice: {row.get('invoice_id')} | Amount: ₹{row.get('amount')} | Category: {row.get('exception_category')}"):
                    st.markdown(f"**Explanation:** {row.get('explanation')}")
                    st.markdown(f"**Recommended Action:** `{row.get('recommended_action')}`")
                    st.markdown(f"**Severity:** `{row.get('severity')}` | **LLM Analyzed:** `{row.get('llm_used', False)}`")
        else:
            st.success("Zero unmatched exceptions in this run!")

    # ── TAB 4: ADVERSARIAL AUDITOR ──────────────────────────────────
    with t4:
        st.subheader("Skeptical LLM Adversarial Audit Log")
        st.info("The Adversarial Auditor independently checks matches for risk factors without altering decision code.")
        if not dec_df.empty and "audit_metadata" in dec_df.columns:
            audit_rows = []
            for _, row in dec_df.iterrows():
                meta = row.get("audit_metadata")
                if meta and isinstance(meta, dict) and meta.get("audit_status"):
                    audit_rows.append({
                        "ERP Invoice": row.get("erp_invoice_id"),
                        "Bank Ref": row.get("bank_ref"),
                        "Suspicious": meta.get("suspicious", False),
                        "Risk Level": meta.get("audit_risk", "LOW"),
                        "Auditor Reason": meta.get("auditor_reason", "N/A"),
                        "Recommendation": meta.get("audit_recommendation", "N/A"),
                        "Final Decision": row.get("final_decision")
                    })
            if audit_rows:
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
            else:
                st.info("No matches triggered secondary adversarial audit.")

    # ── TAB 5: EVALUATION ──────────────────────────────────────────
    with t5:
        st.subheader("Model & Pipeline Evaluation Against Ground Truth")
        if gt_eval and gt_eval.get("precision") is not None:
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Precision", f"{gt_eval.get('precision', 0):.1%}")
            ec2.metric("Recall", f"{gt_eval.get('recall', 0):.1%}")
            ec3.metric("F1 Score", f"{gt_eval.get('f1_score', 0):.1%}")

            st.divider()

            if ml_eval and "pr_curve" in ml_eval:
                pr = ml_eval["pr_curve"]
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR Curve", line=dict(color="#10B981", width=3)))
                fig_pr.update_layout(title="Precision-Recall Curve (XGBoost)", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning("Ground truth labels unavailable for uploaded custom CSVs. Connect ground truth data to view evaluation curves.")

    # ── TAB 6: AUDIT TRAIL ──────────────────────────────────────────
    with t6:
        st.subheader("Tamper-Evident SHA-256 Audit Chain")
        if results["chain_verified"]:
            st.success("🔒 Audit Chain Status: VERIFIED (All SHA-256 hashes unbroken)")
        else:
            st.error(f"⚠️ Audit Chain Status: CORRUPTED at Index {results.get('corrupted_index')}")

        chain_df = results.get("audit_chain_df")
        if chain_df is not None and not chain_df.empty:
            st.dataframe(chain_df, use_container_width=True)

    # ── TAB 7: DOWNLOADS ───────────────────────────────────────────
    with t7:
        st.subheader("Export Reconciliation Results")
        dc1, dc2, dc3 = st.columns(3)

        if not dec_df.empty:
            csv_dec = dec_df.to_csv(index=False).encode("utf-8")
            dc1.download_button("📥 Download Matches CSV", csv_dec, "reconciliation_matches.csv", "text/csv", use_container_width=True)

        if not exc_df.empty:
            csv_exc = exc_df.to_csv(index=False).encode("utf-8")
            dc2.download_button("📥 Download Exceptions CSV", csv_exc, "reconciliation_exceptions.csv", "text/csv", use_container_width=True)

        chain_df = results.get("audit_chain_df")
        if chain_df is not None and not chain_df.empty:
            csv_chain = chain_df.to_csv(index=False).encode("utf-8")
            dc3.download_button("📥 Download Audit Chain CSV", csv_chain, "audit_trail_hash_chain.csv", "text/csv", use_container_width=True)
else:
    st.info("👈 Choose a Demo Scenario or upload CSV files in the sidebar, then click **Run Reconciliation** to begin.")
