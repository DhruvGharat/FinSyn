import hashlib
import json
from datetime import datetime, timezone
import pandas as pd


class AuditRecord:
    def __init__(self, index: int, erp_ref: str, bank_ref: str, gw_ref: str, confidence: float, decision: str, metadata: dict, previous_hash: str, timestamp: str = None):
        self.index = index
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        self.erp_ref = str(erp_ref or "N/A")
        self.bank_ref = str(bank_ref or "N/A")
        self.gw_ref = str(gw_ref or "N/A")
        self.confidence = float(confidence or 0.0)
        self.decision = str(decision or "UNMATCHED")
        self.metadata = metadata or {}
        self.previous_hash = previous_hash
        self.this_hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "erp_ref": self.erp_ref,
            "bank_ref": self.bank_ref,
            "gw_ref": self.gw_ref,
            "confidence": self.confidence,
            "decision": self.decision,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "erp_ref": self.erp_ref,
            "bank_ref": self.bank_ref,
            "gw_ref": self.gw_ref,
            "confidence": self.confidence,
            "decision": self.decision,
            "previous_hash": self.previous_hash,
            "this_hash": self.this_hash,
            "metadata": json.dumps(self.metadata)
        }


class AuditChain:
    def __init__(self):
        self.chain = []
        self._genesis_hash = "0" * 64

    def add_record(self, erp_ref: str, bank_ref: str, gw_ref: str, confidence: float, decision: str, metadata: dict = None) -> AuditRecord:
        previous_hash = self.chain[-1].this_hash if self.chain else self._genesis_hash
        index = len(self.chain)
        record = AuditRecord(
            index=index,
            erp_ref=erp_ref,
            bank_ref=bank_ref,
            gw_ref=gw_ref,
            confidence=confidence,
            decision=decision,
            metadata=metadata,
            previous_hash=previous_hash
        )
        self.chain.append(record)
        return record

    def verify_audit_chain(self) -> tuple:
        """
        Walks the chain and verifies that every SHA-256 hash is unbroken.
        Returns (is_valid: bool, first_corrupted_index: int).
        """
        for i in range(len(self.chain)):
            current = self.chain[i]

            # 1. Verify self hash calculation
            if current.calculate_hash() != current.this_hash:
                return False, i

            # 2. Verify link to previous hash
            if i > 0:
                previous = self.chain[i - 1]
                if current.previous_hash != previous.this_hash:
                    return False, i
            else:
                if current.previous_hash != self._genesis_hash:
                    return False, 0

        return True, -1

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.chain])
