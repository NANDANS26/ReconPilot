# ReconPilot — Demonstration Script & Evaluation Walkthrough

## Phase 01: Foundation Verification Script

This walkthrough demonstrates the verifiable state of the ReconPilot application repository at the conclusion of Phase 01.

### Step 1: Health & Liveness Verification
1. Open terminal and run:
   ```bash
   curl -s http://127.0.0.1:8000/api/v1/health
   ```
2. Verify that output indicates:
   - `status: "healthy"`
   - `service: "reconpilot-backend"`
   - `version: "0.1.0"`
   - `integrations.ai_provider: "mock"`

### Step 2: Honest Dependency Diagnostic Verification
1. Run:
   ```bash
   curl -s http://127.0.0.1:8000/api/v1/health/ready
   ```
2. Note that when the local PostgreSQL database service is not started, the backend responds with HTTP 503 and safe diagnostic information:
   - `database.connected: false`
   - Does NOT fabricate a fake success.
   - Transparently indicates dependency readiness to operational monitoring tools.

### Step 3: Frontend Interface & Connectivity
1. Start frontend development server or preview build:
   ```bash
   cd client
   npm run dev
   ```
2. Navigate to `http://localhost:5173/`.
3. Observe:
   - Deep navy / charcoal financial navigation bar.
   - Live **Backend Status Banner** reporting the real-time health of the FastAPI process.
   - Clean route navigation between Dashboard, Reconciliation, Exceptions, Review Queue, Transactions, Analytics, Audit Trail, and Settings.
   - Honest empty states explicitly identifying upcoming Phase 02–05 implementation rather than fabricated mock numbers.

---

## Planned Subsequent Phase Walkthroughs
- **Phase 02 Demo:** Ingesting raw ledger CSV and synchronizing payment gateway transactions.
- **Phase 03 Demo:** Executing 2-way deterministic reconciliation with fee mismatch detection.
- **Phase 04 Demo:** Viewing structured AI evidence dossiers in the Human Review Queue and approving adjustments.
- **Phase 05 Demo:** Exporting immutable audit logs and viewing match-rate evaluation benchmarks.
