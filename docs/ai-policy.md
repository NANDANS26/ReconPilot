# ReconPilot — AI Investigation & Policy Gate Specification

## 1. Role of AI in Financial Operations
In ReconPilot, Artificial Intelligence functions exclusively as an **Investigative Reasoner**. It operates on bounded, structured evidence packets created by deterministic reconciliation code.

AI does NOT:
- Calculate settlement amounts or balances.
- Alter database tables directly.
- Converse via open-ended chatbots.
- Query unverified external endpoints.

## 2. Evidence Dossier Structure (Planned: Phase 03/04)
When an exception occurs, the system packages evidence into a structured schema:
```json
{
  "exception_id": "EXP-2026-00492",
  "exception_type": "FEE_MISMATCH",
  "discrepancy_amount": 14.50,
  "currency": "INR",
  "ledger_entry": { ... },
  "gateway_entry": { ... },
  "pricing_contract": {
    "standard_rate_pct": 2.0,
    "gst_rate_pct": 18.0
  },
  "historical_resolution_pattern": "Recent gateway surcharge on international cards"
}
```

## 3. Constrained Output Schema
The AI investigation provider (Local ML Classifier or Mock Provider) must respond matching a strict Pydantic model:
```json
{
  "hypothesized_cause": "International card processing fee tier applied rather than domestic debit slab.",
  "suggested_classification": "FEE_SURCHARGE_INTERNATIONAL",
  "confidence_score": 0.94,
  "recommended_action": "RECONCILE_WITH_FEE_ADJUSTMENT",
  "evidence_rationale": "Card BIN 411111 resolves to non-domestic issuing bank subject to 3.0% agreed schedule."
}
```

## 4. Policy Gate Architecture
The policy engine acts as a guardrail before applying actions:
- **Auto-Resolve Gate:** Confidence $\ge$ `AI_AUTO_CLASSIFY_THRESHOLD` (default 0.90) AND amount $\le$ ₹ 5,000.
- **Review Queue Gate:** Confidence $<$ `AI_AUTO_CLASSIFY_THRESHOLD` OR amount $>$ ₹ 5,000 $\rightarrow$ automatically routed to Human Finance Controller Queue.
- **Rejection Gate:** Invalid JSON schema or confidence $<$ `AI_REVIEW_THRESHOLD` (0.75) $\rightarrow$ flagged as unclassified exception requiring manual investigation.
