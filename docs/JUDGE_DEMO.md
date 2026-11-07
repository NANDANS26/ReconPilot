# ReconPilot — 5–7 Minute Evaluation Demonstration Script

> **Product:** ReconPilot Autonomous AI Finance Controller
> **Demonstration Target Duration:** 5 to 7 minutes  
> **Guiding Principle:** Deterministic financial truth first, evidence-grounded local ML reasoning second, dual-control human governance always. Zero automated ledger mutations.

---

## Pre-Flight Checklist (1 Minute Before Demo)

1. **Verify Backend Status**: Open [http://127.0.0.1:8000/api/v1/health/ready](http://127.0.0.1:8000/api/v1/health/ready). Confirm `"status": "ready"` and PostgreSQL is connected.
2. **Open Frontend**: Navigate to [http://localhost:5173/](http://localhost:5173/).
3. **Verify Demo Banner**: Confirm the top amber banner displays: `"Judge Demo Environment — Seeded Demonstration Records"`.
4. **Login Credentials**:
   - **Finance Operator**: `operator@reconpilot.local` / `OperatorSecret2026!` (Can propose resolutions).
   - **Finance Reviewer / Lead**: `reviewer@reconpilot.local` / `ReviewerSecret2026!` (Can approve/reject resolutions).
   - **System Admin**: `admin@reconpilot.local` / `AdminSecret2026!`

---

## Act 1 — The Problem & Command Center (1 Minute)

### Where to Navigate
- Open [http://localhost:5173/dashboard](http://localhost:5173/dashboard) (Finance Operations Command Center).

### What to Show
1. Top **Executive KPI Cards**:
   - **Payments Processed**: Internal merchant ledger count.
   - **Deterministic Match Rate**: E.g., `96.8%` (matched vs exceptional).
   - **Outstanding Financial Exposure**: Currency-separated chips (e.g., `INR: ₹48,000.00`, `USD: $1,250.00`).
2. **Reconciliation Health Visualizations**:
   - Donut chart showing match distributions.
   - Horizontal bar chart classifying exception types.
   - 14-Day Reconciliation Activity & Exposure trendline.

### What to Explain
> *"High-growth digital businesses process millions of transactions across internal databases, payment gateways, and partner banks. Spreadsheets and brittle scripts fail when fees drift or settlement files arrive late. Furthermore, finance teams cannot trust generative AI to calculate financial balances or mutate ledgers.*
> 
> *ReconPilot solves this by enforcing a hard separation of concerns: mathematical reconciliation is strictly deterministic in PostgreSQL, while AI is used strictly as an assistive diagnostics investigator. Notice that our exposure totals are segregated by currency (INR vs USD) — we never sum different currencies together."*

### Key Judge Takeaway
- Deterministic metrics and multi-currency safety are enforced at the database level.

---

## Act 2 — Deterministic Reconciliation & Discrepancy Classification (1 Minute)

### Where to Navigate
- In the Command Center, locate the **Demo Scenario Selector** card or navigate to `/exceptions`.
- Click **"Explore"** on **Discrepancy: Amount Mismatch** (e.g., Exception `439BC332`).

### What to Show
1. The **Financial Discrepancy Calculation** header card:
   - **Expected Settlement Amount**: ₹4,900.00
   - **Actual Processor Credit**: ₹4,400.00
   - **Discrepancy Variance**: ₹-500.00
2. Deterministic formula:  
   $$\text{Expected Settlement} = \text{Gross Payment} - \text{Refunds} - \text{Fees} + \text{Adjustments}$$

### What to Explain
> *"Every single calculation here is computed using 28-digit fixed-point Decimal arithmetic in Python and PostgreSQL NUMERIC(20, 2). Notice that no LLM was involved in detecting this difference. The deterministic engine compared the internal order with the settlement batch line, calculated the expected payout, identified the ₹500 variance, and classified it as an `AMOUNT_MISMATCH`."*

### Key Judge Takeaway
- No generative AI hallucinations or floating-point errors in arithmetic or matching.

---

## Act 3 — Correlated Evidence Dossier & Provenance (1 Minute)

### Where to Navigate
- Scroll down on the Exception Detail page to the **Correlated Financial Evidence Dossier** card.

### What to Show
1. **Corroborated Facts** list (e.g., payment captured, batch timestamp confirmed).
2. **Missing Information & Direct Conflicts** panels.
3. The **Correlated Evidence Items** table:
   - Expand an item (e.g., `Gateway Payment Record`, `Processor Settlement Line`).
   - Show the **Evidence Strength Badge** (`CONFIRMED`, `CORROBORATED`, `CONFLICTING`).
   - Show the JSON raw payload drawer detailing external IDs, timestamps, and fee schedules.

### What to Explain
> *"Before an investigation begins, our Phase 05 Evidence Correlator aggregates all related audit records into an immutable evidence dossier. It flags corroborated facts, detects missing data, and highlights contradictory records. Every single finding links directly back to a primary key in our database."*

### Key Judge Takeaway
- Complete provenance and audit trail grounding for every financial variance.

---

## Act 4 — AI Root Cause Investigation (1.5 Minutes)

### Where to Navigate
- Scroll to the **AI Root Cause Investigation Service** card.

### What to Show
1. The **AI Provider State Badge**:
   - Explicitly displays: `Local ML Intelligence (reconpilot-catboost-champion)` (or `Deterministic Mock Provider` in fallback mode).
2. The **Evaluated Hypotheses & Causes**:
   - Each hypothesis displays:
     - Root cause explanation.
     - Evidence grounding assessment (`SUPPORTED`, `PLAUSIBLE`, or `UNDETERMINED`).
     - **Citing Evidence badges** (e.g., `citing: EV-PAY-01`, `EV-SETTLE-02`).
3. **Recommended Operational Actions**:
   - Specific advisory steps (e.g., *"Initiate processor inquiry for missing ₹500 fee deduction"*).
   - Prominent badge: `"Review Required"`.
4. **Provenance & Limitations Footer**:
   - Provider name, model name, feature schema version, and cryptographic evidence snapshot hash.

### What to Explain
> *"Now we introduce local AI — but with strict constraints. The reasoner operates exclusively on the verified evidence dossier. It cannot hallucinate external data because the validator rejects any hypothesis that references non-existent evidence IDs.*
> 
> *Notice our transparency badge: it clearly states 'Local ML Intelligence (reconpilot-catboost-champion)'. ReconPilot runs an embedded, offline CatBoost classifier with calibrated confidence scoring and margin abstention gates. Most importantly, AI output is strictly advisory — all operational recommendations require human approval."*

### Key Judge Takeaway
- Schema-constrained, evidence-cited AI diagnostics with full provider transparency and zero automated mutations.

---

## Act 5 — Dual-Control Human Approval Governance (1 Minute)

### Where to Navigate
- Scroll down to the **Human Approval & Exception Resolution Workflow** card.

### What to Show
1. **Resolution Proposal Panel**:
   - Show available resolution categories: `MANUAL_MATCH`, `WRITE_OFF`, `ESCALATE_TO_GATEWAY`, `FORCE_CLOSE`.
   - Submit a proposal with rationale (as `operator@reconpilot.local`).
2. **Separation of Duties (SoD) Enforcement**:
   - Show the proposal appear in the audit trail.
   - Point out that the proposer **cannot approve their own proposal**:
     - The approval button displays a lock icon: `"Requires different approver (Separation of Duties)"`.
3. *(Optional/Two-Account Demo)*:
   - Log out, log in as `reviewer@reconpilot.local` (`ReviewerSecret2026!`).
   - The approval controls become enabled with green **"Approve Proposal"** and red **"Reject Proposal"** buttons.
4. **Resolution Audit History**:
   - An immutable chronological log recording every submission, decision, and operator comment.

### What to Explain
> *"In enterprise finance, automated ledger writes are a compliance violation. ReconPilot enforces dual-control approval. An operator can propose an action (such as writing off ₹500 or escalating to the payment processor), but our Separation of Duties engine prevents self-approval. A separate reviewer or finance lead must review the evidence dossier and make the recorded decision."*

### Key Judge Takeaway
- Strict financial controls: dual-control governance and separation of duties prevent fraud and unilateral write-offs.

---

## Act 6 — Operations Queue & Return to Command Center (30 Seconds)

### Where to Navigate
- Return to [http://localhost:5173/dashboard](http://localhost:5173/dashboard).
- Scroll down to the **Operations Queue** table.

### What to Show
1. The paginated exception queue with live search and multi-filtering (Status, Exception Type, Sort by Financial Impact).
2. The proposal status badge reflecting the proposal submitted in Act 5.
3. Summary of how this closed-loop workflow resolves operational bottlenecks for finance teams.

### Closing Statement
> *"To summarize: ReconPilot delivers deterministic mathematical accuracy, correlated evidence dossiers, transparent assistive AI root-cause diagnostics, and dual-control human sign-off — all reflected in a real-time finance operations command center. Thank you!"*

---

## Demonstration Fallback Strategies

| Scenario / Blocker | Fallback Action During Live Demo |
|:---|:---|
| **Local ML model artifact missing** | The application automatically falls back to the deterministic rule-grounded mock provider and badges it as `Deterministic Mock Provider`. Explain that the mock provider enforces the exact same Pydantic schema validation without degraded user experience. |
| **Two-account login takes too long** | Stay logged in as `operator@reconpilot.local`. Show the disabled approval button and point out the lock badge: `"Requires different approver (Separation of Duties)"`. Explain this is proof that SoD is working. |
| **Specific scenario missing** | The Demo Scenario Selector dynamically disables scenarios not found in the current dataset. Simply click any scenario labeled with a green `Available` badge. |
| **Accidental navigation away** | Click the floating **Judge Walkthrough Guide** in the bottom-right corner and click `"Open Feature"` to instantly jump back to the intended step. |
