import os
import pandas as pd
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM with fallback safety check
GROQ_KEY = os.getenv("GROQ_API_KEY")
llm_available = bool(GROQ_KEY and GROQ_KEY.strip() and not GROQ_KEY.startswith("your_"))

if llm_available:
    try:
        llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0)
    except Exception:
        llm = None
        llm_available = False
else:
    llm = None


# ── SHARED STATE ─────────────────────────────────────────────
class AgentState(TypedDict):
    medium_pairs:    list   # pairs needing LLM review
    high_pairs:      list   # pairs for adversarial audit
    exceptions:      list   # unmatched rows
    explained:       list   # exception explanations from Node 1
    audited:         list   # audit results from Node 2
    final_matches:   list   # confirmed matches after all checks


# ── NODE 1: EXCEPTION EXPLAINER ───────────────────────────────
class ExceptionExplanation(BaseModel):
    category: Literal[
        "FX_ROUNDING",
        "TIMING_LAG",
        "REF_MISMATCH",
        "SPLIT_PAYMENT",
        "MERGED_PAYMENT",
        "MISSING_GATEWAY_RECORD",
        "MISSING_BANK_RECORD",
        "MISSING_ERP_RECORD",
        "CURRENCY_MISMATCH",
        "DEBIT_CREDIT_MISMATCH",
        "OTHER"
    ]
    explanation: str = Field(description="Clear explanation of why record is unmatched")
    recommended_action: str = Field(description="Recommended action for finance analyst")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Risk severity level")


explainer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a financial reconciliation expert explaining unmatched records.
     Categories:
     - FX_ROUNDING: minor amount difference (< 1%) due to fees or FX rounding
     - TIMING_LAG: date settlement lag beyond standard window
     - REF_MISMATCH: reference numbers do not match
     - SPLIT_PAYMENT: invoice split into multiple payments
     - MERGED_PAYMENT: multiple invoices paid in single lump sum
     - MISSING_GATEWAY_RECORD: record present in ERP but missing in gateway
     - MISSING_BANK_RECORD: record present in ERP but missing in bank
     - MISSING_ERP_RECORD: bank transaction has no ERP invoice
     - CURRENCY_MISMATCH: different currency codes
     - DEBIT_CREDIT_MISMATCH: transaction debit/credit direction conflict
     - OTHER: general discrepancy

     Return valid JSON matching the specified output schema."""),
    ("human",
     """Unmatched ERP record:
     Invoice ID:   {invoice_id}
     Amount:       ₹{amount}
     Date:         {date}
     Customer:     {customer_id}
     Description:  {description}

     Closest bank candidate considered:
     Bank Ref:     {closest_bank_ref}
     Bank Amount:  ₹{closest_bank_amount}
     Bank Date:    {closest_bank_date}
     Cost Score:   {closest_cost}""")
])

if llm_available and llm:
    explainer_chain = explainer_prompt | llm.with_structured_output(ExceptionExplanation)
else:
    explainer_chain = None


def deterministic_fallback_explainer(exc: dict) -> dict:
    """Deterministic fallback categorization when LLM is unavailable."""
    amt = float(exc.get("amount", 0.0))
    closest_amt_str = str(exc.get("closest_bank_amount", "N/A"))

    try:
        closest_amt = float(closest_amt_str)
        amt_diff = abs(amt - closest_amt)
        pct_diff = (amt_diff / max(amt, 1.0)) * 100.0
    except ValueError:
        amt_diff = 999.0
        pct_diff = 100.0

    if pct_diff < 1.0 and amt_diff > 0:
        cat = "FX_ROUNDING"
        exp = f"Minor amount discrepancy of ₹{amt_diff:.2f} ({pct_diff:.2f}%) likely due to gateway fees or rounding."
        action = "Approve small tolerance adjustment"
        sev = "LOW"
    elif exc.get("closest_bank_ref") == "None" or closest_amt_str == "N/A":
        cat = "MISSING_BANK_RECORD"
        exp = "No matching bank settlement record found within date window."
        action = "Check with gateway settlement team for pending payout"
        sev = "HIGH"
    elif pct_diff > 15.0:
        cat = "REF_MISMATCH"
        exp = f"Reference mismatch and significant amount difference of ₹{amt_diff:.2f}."
        action = "Manual reconciliation required"
        sev = "MEDIUM"
    else:
        cat = "TIMING_LAG"
        exp = "Possible date lag or settlement window mismatch."
        action = "Review bank statement dates"
        sev = "LOW"

    return {
        **exc,
        "exception_category": cat,
        "explanation": f"[Deterministic Rule] {exp}",
        "recommended_action": action,
        "severity": sev,
        "llm_used": False
    }


def run_exception_explainer(state: AgentState) -> AgentState:
    exceptions = state.get("exceptions", [])
    explained = []

    # Limit LLM requests to max 10 exceptions for speed/performance
    max_llm_calls = 10
    llm_call_count = 0

    for exc in exceptions:
        if llm_available and explainer_chain and llm_call_count < max_llm_calls:
            try:
                res = explainer_chain.invoke({
                    "invoice_id":          exc.get("invoice_id", "N/A"),
                    "amount":              exc.get("amount", "N/A"),
                    "date":                exc.get("date", "N/A"),
                    "customer_id":         exc.get("customer_id", "N/A"),
                    "description":         exc.get("description", "N/A"),
                    "closest_bank_ref":    exc.get("closest_bank_ref", "None"),
                    "closest_bank_amount": exc.get("closest_bank_amount", "N/A"),
                    "closest_bank_date":   exc.get("closest_bank_date", "N/A"),
                    "closest_cost":        exc.get("closest_cost", "N/A"),
                })
                llm_call_count += 1
                explained.append({
                    **exc,
                    "exception_category": res.category,
                    "explanation":        res.explanation,
                    "recommended_action": res.recommended_action,
                    "severity":           res.severity,
                    "llm_used":           True
                })
                continue
            except Exception:
                pass

        # Fallback if LLM fails, capped, or unavailable
        explained.append(deterministic_fallback_explainer(exc))

    return {**state, "explained": explained}


# ── NODE 2: ADVERSARIAL AUDITOR ───────────────────────────────
class AdversarialAuditReport(BaseModel):
    suspicious: bool = Field(description="True if auditor finds potential mismatch risk")
    risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Assessed risk level")
    reason: str = Field(description="Skeptical reasoning explaining potential risk")
    recommendation: Literal["AUTO_ACCEPT", "MANUAL_REVIEW", "REJECT"] = Field(
        description="Recommended handling queue"
    )


auditor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a skeptical financial auditor inspecting high/medium confidence payment matches.
     Actively check for:
     - Suspicious round amounts shared by multiple transactions
     - Missing customer or reference ID links
     - Date coincidences without matching reference strings
     - Inconsistent customer info

     Return structured JSON with: suspicious (bool), risk ('LOW'|'MEDIUM'|'HIGH'), reason (str), recommendation ('AUTO_ACCEPT'|'MANUAL_REVIEW'|'REJECT').
     IMPORTANT: Do NOT decide match status directly; provide audit risk analysis."""),
    ("human",
     """Proposed Match:
     ERP Invoice: {erp_id} | Amount: ₹{erp_amount} | Date: {erp_date}
     Bank Ref:    {bank_ref} | Amount: ₹{bank_amount} | Date: {bank_date}
     ML Confidence: {confidence} | Amount Diff: ₹{amt_diff} | Ref Similarity: {ref_sim}

     Perform skeptical audit.""")
])

if llm_available and llm:
    auditor_chain = auditor_prompt | llm.with_structured_output(AdversarialAuditReport)
else:
    auditor_chain = None


def run_adversarial_auditor(state: AgentState) -> AgentState:

    pairs_to_audit = state.get("medium_pairs", []) + [
        p for p in state.get("high_pairs", [])
        if float(p.get("erp_amount", 0)) % 1000 == 0
    ]
    audited = []

    max_llm_calls = 10
    llm_call_count = 0

    for pair in pairs_to_audit:
        if llm_available and auditor_chain and llm_call_count < max_llm_calls:
            try:
                res = auditor_chain.invoke({
                    "erp_id":      pair.get("erp_invoice_id", "N/A"),
                    "erp_amount":  pair.get("erp_amount", "N/A"),
                    "erp_date":    pair.get("erp_date", "N/A"),
                    "bank_ref":    pair.get("bank_ref", "N/A"),
                    "bank_amount": pair.get("bank_amount", "N/A"),
                    "bank_date":   pair.get("bank_date", "N/A"),
                    "confidence":  pair.get("confidence", "N/A"),
                    "amt_diff":    round(abs(float(pair.get("erp_amount", 0)) - float(pair.get("bank_amount", 0))), 2),
                    "ref_sim":     pair.get("ref_similarity", "N/A"),
                })
                llm_call_count += 1
                audited.append({
                    **pair,
                    "suspicious":     res.suspicious,
                    "audit_risk":     res.risk,
                    "auditor_reason": res.reason,
                    "audit_recommendation": res.recommendation,
                    "audit_status":   "AUDITED"
                })
                continue
            except Exception:
                pass

        # Deterministic audit fallback
        erp_amt = float(pair.get("erp_amount", 0))
        is_round = (erp_amt > 0 and erp_amt % 1000 == 0)
        ref_sim = float(pair.get("ref_similarity", 0))

        if is_round and ref_sim < 0.3:
            susp = True
            risk = "MEDIUM"
            reason = "Round amount transaction with low reference similarity."
            rec = "MANUAL_REVIEW"
        else:
            susp = False
            risk = "LOW"
            reason = "No skeptical risk flags raised by deterministic auditor rules."
            rec = "AUTO_ACCEPT"

        audited.append({
            **pair,
            "suspicious": susp,
            "audit_risk": risk,
            "auditor_reason": f"[Deterministic Rule] {reason}",
            "audit_recommendation": rec,
            "audit_status": "UNAVAILABLE" if not llm_available else "FALLBACK"
        })

    return {**state, "audited": audited}



# ── NODE 3: ASSEMBLER ─────────────────────────────────────────
def run_assembler(state: AgentState) -> AgentState:
    """
    Combines ML matching results and LLM audit risk flags.
    DETERMINISTIC LOGIC decides final decision queue based on confidence and audit risk.
    """
    final = []
    audited_map = { (p["erp_invoice_id"], p["bank_ref"]): p for p in state.get("audited", []) }

    for pair in state.get("high_pairs", []):
        key = (pair["erp_invoice_id"], pair["bank_ref"])
        audit_info = audited_map.get(key)

        if audit_info and audit_info.get("suspicious") and audit_info.get("audit_risk") == "HIGH":
            decision = "MANUAL_REVIEW"
            status = "AUDIT_FLAGGED_HIGH_RISK"
        else:
            decision = "AUTO_ACCEPTED"
            status = "AUTO_MATCHED"

        final.append({
            **pair,
            "final_decision": decision,
            "final_status": status,
            "audit_metadata": audit_info if audit_info else {}
        })

    for pair in state.get("medium_pairs", []):
        key = (pair["erp_invoice_id"], pair["bank_ref"])
        audit_info = audited_map.get(key, pair)

        rec = audit_info.get("audit_recommendation", "MANUAL_REVIEW")
        if rec == "AUTO_ACCEPT" and not audit_info.get("suspicious"):
            decision = "REVIEW_APPROVED"
        elif rec == "REJECT":
            decision = "AUTO_REJECTED"
        else:
            decision = "MANUAL_REVIEW"

        final.append({
            **pair,
            "final_decision": decision,
            "final_status": f"AUDIT_{rec}",
            "audit_metadata": audit_info
        })

    return {**state, "final_matches": final}