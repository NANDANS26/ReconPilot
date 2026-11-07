# ReconPilot — Autonomous Financial Reconciliation & Exception Investigation

> **ReconPilot AI Finance Controller**
> **Core Guarantee:** Purely deterministic financial matching, locally-intelligent ML root-cause reasoning, cryptographically-hashed evidence snapshots, and dual-control human governance. Zero automated ledger mutations.

---

## 1. Executive Summary & Architecture

Modern digital enterprises process millions of transactions across internal order systems, payment gateways, and banking networks. When fees drift, refunds arrive out of order, or settlement batches lag, manual spreadsheet reconciliation breaks down and financial leakage goes unnoticed.

Conversely, naive generative AI approaches present severe financial and operational risks: hallucinations in arithmetic, fabricated transaction balances, non-deterministic reasoning, or unauthorized ledger mutations.

**ReconPilot resolves this dilemma through a strict separation of concerns:**

1. **Deterministic Financial Truth:** All mathematical calculations, matching algorithms, and balance calculations are executed deterministically using fixed-point `Decimal` arithmetic in Python and PostgreSQL ACID transactional storage.
2. **Correlated Evidence Provenance & Cryptographic Snapshots:** Prior to diagnostic analysis, an immutable evidence dossier is correlated from source records. Every dossier is canonicalized and sealed with a deterministic SHA-256 hash, ensuring tamper-evident provenance.
3. **Local Machine Learning Root-Cause Classifier:** Exception root-cause hypotheses are evaluated locally via an embedded CatBoost champion pipeline (rigorously benchmarked across Random Forest, HistGradientBoosting, XGBoost, LightGBM, and soft-voting ensembles). The ML engine operates entirely on integer minor-unit financial features (paise/cents) without target leakage or external network calls. Calibrated probabilities, conservative confidence thresholding, and margin-based abstention protect against ambiguity.
4. **Evidence-Grounded Explanations:** Explanations cite only verified evidence IDs from the canonical snapshot and enforce strict policy gates (e.g. mandatory human review on high financial exposure).
5. **Dual-Control Human Governance:** Operational resolutions (such as fee credit requests, invoice adjustments, or write-offs) are submitted as structured proposals. An automated Separation of Duties (SoD) engine prevents self-approval, requiring authorized reviewer sign-off and maintaining an immutable audit history.

---

## 2. Complete Product Workflow

```
 SOURCE RECORDS (Internal Ledgers, Payment Gateway Feeds, Bank Statements)
                     │
                     ▼
 DETERMINISTIC MATCHING ENGINE (Exact 2-Way, Bilateral Invariant Matching, Duplicate Quarantine)
                     │
                     ├── [Matched Records] ──► Expected Settlement Verified
                     │
                     └── [Discrepancies] ────► Categorized Financial Exceptions
                                                   │
                                                   ▼
                                     EVIDENCE CORRELATION & SNAPSHOTS
                                     (Dossier Assembly & SHA-256 Hash)
                                                   │
                                                   ▼
                                     LOCAL ML ROOT-CAUSE REASONER
                                     (Embedded Pipeline, Abstention Gates)
                                                   │
                                                   ▼
                                     DUAL-CONTROL HUMAN APPROVAL
                                     (Proposals, SoD, Decision Log)
                                                   │
                                                   ▼
                                     COMMAND CENTER TELEMETRY
```

---

## 3. Technology Stack & Key Highlights

| Layer | Technologies | Architectural Invariants |
|:---|:---|:---|
| **Frontend** | React 18, TypeScript (Strict), Vite, Tailwind CSS, Recharts, Lucide Icons | Premium fintech dashboard with real-time KPI metrics, dark aesthetic, responsive charts, and guided demo experience. |
| **Backend API**| Python 3.11+ / FastAPI, Pydantic 2.x, Uvicorn ASGI | Strict API contracts, asynchronous I/O, RFC-compliant JWT authorization, and comprehensive endpoint validation. |
| **Database** | **PostgreSQL (Exclusively)** with SQLAlchemy 2.x & Alembic | ACID transactional guarantees, row-level locking, additive schema migrations, and `NUMERIC(20, 2)` monetary precision. |
| **Reconciliation**| Python `Decimal` fixed arithmetic (28-digit precision) | Zero floating-point drift, bilateral 1-to-1 candidate matching, and input permutation invariance. |
| **ML Engine** | CatBoost, LightGBM, XGBoost, scikit-learn, joblib | Offline multiclass CatBoost Champion (with multi-model benchmarking suite, versioned schema `v1.0`, Platt scaling calibration, and conservative margin abstention). |
| **Security & Auditing** | Argon2id, Cryptographic SHA-256, RBAC | Tamper-evident evidence hashing, multi-tenant isolation, and complete audit trail. |

---

## 4. Local Machine Learning Engine

ReconPilot replaces brittle cloud LLM prompts with an embedded, auditable, offline machine learning pipeline:

- **Feature Schema (`v1.0`):** 31 numeric features in integer minor units (paise/cents) + 5 categorical features. Zero target leakage: all UUIDs, entity codes, and ground-truth answer labels are strictly excluded.
- **Root-Cause Classification:** Classifies exceptions across 11 financial root-cause classes:
  - `FEE_MISMATCH`
  - `REFUND_MISMATCH`
  - `ADJUSTMENT_MISMATCH`
  - `AMOUNT_MISMATCH`
  - `MISSING_PAYMENT`
  - `MISSING_SETTLEMENT`
  - `DUPLICATE_PAYMENT`
  - `DUPLICATE_SETTLEMENT`
  - `DELAYED_SETTLEMENT`
  - `AMBIGUOUS_MATCH`
  - `INSUFFICIENT_EVIDENCE`
- **Conservative Abstention Gates:** When the predicted class probability falls below `ML_CONFIDENCE_THRESHOLD` (default: 0.50) or the difference between top candidate probabilities is less than `ML_MIN_MARGIN` (default: 0.10), the model safely abstains to `INSUFFICIENT_EVIDENCE` or `AMBIGUOUS_MATCH` and requires human operator review.
- **Model Training, Multi-Algorithm Benchmark & Validation:**
  ```bash
  # Run full 5-algorithm benchmarking suite with Platt calibration and champion selection:
  python -m app.ml.benchmarking.runner

  # Run final independent evaluation and adversarial stress validation:
  python -m app.ml.final_independent_validation

  # Run held-out evaluation and confusion matrix inspection:
  python -m app.ml.evaluation
  ```

---

## 5. Local Setup & Startup Guide

### Prerequisites
- **Python:** 3.11 or newer (Python 3.14 verified)
- **Node.js:** v20 or newer and npm
- **PostgreSQL:** 15, 16, 17, or 18 running locally on port 5432

### Step 1: Configure Environment
```bash
# Copy configuration template
cp .env.example .env
```
*(Verify `.env` has your local PostgreSQL credentials in `DATABASE_URL`).*

### Step 2: Backend Setup
```bash
# 1. Activate Python environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install backend dependencies
pip install -e ".[dev]"

# 3. Apply database migrations
cd server
python -m alembic upgrade head
cd ..

# 4. Start the FastAPI backend server
python -m uvicorn app.main:app --app-dir server --reload --host 127.0.0.1 --port 8000
```
- API Server: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Readiness Probe: [http://127.0.0.1:8000/api/v1/health/ready](http://127.0.0.1:8000/api/v1/health/ready)

### Step 3: Frontend Setup
```bash
# In a new terminal window:
cd client
npm install
npm run dev
```
- Frontend Application: [http://localhost:5173](http://localhost:5173)

---

## 6. Seeded Demo Accounts

The local development database includes pre-seeded demonstration accounts for evaluating role-based governance:

| Role | Email | Password | Permissions |
|:---|:---|:---|:---|
| **Finance Operator** | `operator@reconpilot.local` | `OperatorSecret2026!` | Can triage exceptions, run investigations, and submit resolution proposals. |
| **Finance Reviewer** | `reviewer@reconpilot.local` | `ReviewerSecret2026!` | Can review evidence and approve or reject proposals submitted by others (SoD enforced). |
| **System Admin** | `admin@reconpilot.local` | `AdminSecret2026!` | Full administrative access, tenant settings, and user management. |

---

## 7. Verification & Test Suite

ReconPilot maintains exhaustive test coverage across all domain modules:

```bash
# Run complete backend pytest suite (117 passing tests):
python -m pytest server/tests/ -v

# Run Python linter & code formatter:
python -m ruff check server
python -m ruff format server --check

# Run frontend production build verification:
cd client
npm run build
```

---

## 8. Financial Invariants & Security Architecture

1. **Non-Negotiable Determinism:** All arithmetic (fees, refunds, net payouts, variances) is calculated via Python `Decimal` and PostgreSQL `NUMERIC(20, 2)`. No machine learning model or external service is ever used for mathematical calculations.
2. **Order Invariance & Bilateral Uniqueness:** Shuffling the order of payments or settlement items produces bitwise identical matches and exceptions. Candidates are matched if and only if uniquely supported on both sides.
3. **Preventive Duplicate Quarantine:** Duplicate external reference IDs are isolated prior to matching, preventing double-counting or inflated match statistics.
4. **Separation of Duties (SoD):** A user who submits a resolution proposal is strictly prohibited from approving it. Approval requires a distinct reviewer account.
5. **Multi-Currency Protection:** Financial exposures are segregated by currency (e.g. INR vs USD). Cross-currency arithmetic is strictly forbidden.
6. **Zero Automated Ledger Mutations:** Approval records an authorized human decision in the immutable audit trail; it never automatically modifies source ledger records without external human intervention.
