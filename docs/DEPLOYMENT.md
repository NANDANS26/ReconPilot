# ReconPilot — Production & Cloud Deployment Architecture Guide

> **Architecture:** Clean Modular Monolith  
> **Backend:** Python 3.11+ / FastAPI (Uvicorn ASGI)  
> **Frontend:** React 18 / TypeScript SPA (Vite static build)  
> **Database:** Managed PostgreSQL 15+ (ACID compliant)

---

## 1. Production Architecture Overview

```
                      ┌────────────────────────────┐
                      │    Cloudflare / CDN / WAF  │
                      └─────────────┬──────────────┘
                                    │ HTTPS (443)
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   ┌───────────────────────┐                   ┌───────────────────────┐
   │ Static Asset Host     │                   │ Reverse Proxy / ALB   │
   │ (S3+CloudFront, Vercel│                   │ (Nginx / AWS ALB)     │
   │  Cloudflare Pages)    │                   └──────────┬────────────┘
   │ [React 18 Production] │                              │ Reverse Proxy
   └───────────────────────┘                              ▼
                                               ┌───────────────────────┐
                                               │ FastAPI ASGI Cluster  │
                                               │ (Uvicorn / Gunicorn)  │
                                               │ [Docker / Cloud Run]  │
                                               └──────────┬────────────┘
                                                          │ Connection Pool
                                                          ▼ (psycopg3)
                                               ┌───────────────────────┐
                                               │ Managed PostgreSQL    │
                                               │ (RDS / Cloud SQL)     │
                                               │ [ACID Relational Core]│
                                               └───────────────────────┘
```

---

## 2. Infrastructure Requirements

### A. Managed PostgreSQL (Only Database)
- **Version:** PostgreSQL 15, 16, 17, or 18.
- **Hardware Specs (Production Baseline):**
  - Minimum: 2 vCPUs, 4 GB RAM, SSD with automatic backups enabled.
  - Scale: Multi-AZ read replicas for heavy analytical queries if needed.
- **Extensions:** Standard PostgreSQL built-ins (`uuid-ossp`, `pgcrypto`).
- **Connection String:**
  ```text
  postgresql+psycopg://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:<DB_PORT>/<DB_NAME>?sslmode=require
  ```
- **Connection Pooling:**
  - Configured in FastAPI via SQLAlchemy: `DB_POOL_SIZE=10`, `DB_MAX_OVERFLOW=20`.
  - For large scale (100+ concurrent app instances), introduce PgBouncer in transaction pooling mode.

### B. Backend Hosting (FastAPI ASGI)
- **Environment:** Container runtime (Docker, AWS ECS/Fargate, Google Cloud Run) or Linux VM (Ubuntu 22.04/24.04).
- **Process Manager:** `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app`.
- **Health Check Endpoints:**
  - Liveness probe: `GET /api/v1/health` (HTTP 200).
  - Readiness probe: `GET /api/v1/health/ready` (HTTP 200 when PostgreSQL is responsive; HTTP 503 if unreachable).

### C. Frontend Hosting (React Static SPA)
- **Environment:** Any high-performance edge static host (Vercel, AWS S3 + CloudFront, Cloudflare Pages, Nginx container).
- **Build Output:** Static assets produced by `npm run build` in `client/dist/`.
- **SPA Routing Fallback:** All non-file URI routes must fall back to `/index.html` (HTTP 200).

---

## 3. Production Environment Configuration

All environment variables must be securely injected via your cloud platform's secret manager (e.g. AWS Secrets Manager, GCP Secret Manager, Vault).

| Variable | Required | Description | Production Example |
|:---|:---:|:---|:---|
| `APP_ENV` | **Yes** | Runtime environment identifier | `production` |
| `DATABASE_URL` | **Yes** | Managed PostgreSQL connection string | `postgresql+psycopg://dbuser:xxx@db.internal:5432/reconpilot?sslmode=require` |
| `APP_SECRET_KEY` | **Yes** | Server cryptographic secret (min 32 bytes) | Generated 64-char hex string |
| `JWT_SECRET_KEY` | **Yes** | Secret used to sign JWT auth tokens | Generated 64-char hex string |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | **Yes** | Auth session validity period | `30` |
| `CORS_ORIGINS` | **Yes** | Allowed web origins (strict domain match) | `https://reconpilot.yourdomain.com` |
| `API_HOST` | **Yes** | Bind host | `0.0.0.0` |
| `API_PORT` | **Yes** | Bind port | `8000` |
| `VITE_API_BASE_URL` | **Yes** | Client build-time backend API URL | `https://api.reconpilot.yourdomain.com/api/v1` |
| `AI_PROVIDER` | No | Reasoner engine: `local_ml` or `mock` | `local_ml` |
| `ML_MODEL_NAME` | No | Local ML model name | `reconpilot-rf-v1` |
| `ML_CONFIDENCE_THRESHOLD` | No | Minimum confidence for non-abstention | `0.50` |
| `ML_MIN_MARGIN` | No | Minimum probability margin between top classes | `0.10` |

---

## 4. Database Migration Procedure (Zero Data Loss)

Alembic migrations must be executed during the deployment pipeline before new app instances start taking traffic.

```bash
# 1. Inspect current schema revision
alembic -c server/alembic.ini current

# 2. Upgrade schema forward to latest head
alembic -c server/alembic.ini upgrade head

# 3. Confirm head revision matches code
alembic -c server/alembic.ini current
```

### Safety Rules:
- **Never run `alembic downgrade` or database reset scripts in production.**
- All Alembic migration revisions must be reviewed to ensure non-destructive DDL (`ADD COLUMN`, `CREATE TABLE`, `CREATE INDEX`). No `DROP TABLE` or `DROP COLUMN` in production migrations without phased deprecation.

---

## 5. Security & Network Hardening

1. **HTTPS Enforcement:**
   - Terminate TLS 1.3 at Cloudflare/ALB with modern cipher suites.
   - Set HSTS header: `Strict-Transport-Security: max-age=31536000; includeSubDomains`.
2. **CORS Restrictions:**
   - Set `CORS_ORIGINS` strictly to your production frontend hostname. Never use `*` with credentials.
3. **Database Network Isolation:**
   - PostgreSQL must live inside a private VPC with no public IP address. Only backend application instances and authorized bastion hosts can connect on port 5432.
4. **Separation of Duties & Least Privilege:**
   - Role-Based Access Control (`ADMIN`, `FINANCE_CONTROLLER`, `REVIEWER`, `FINANCE_OPERATOR`) is enforced on every API route via FastAPI dependency injection.
   - Proposals can never be approved by their creator; this check is enforced at the database transaction layer.

---

## 6. Logging, Monitoring & Observability

- **Structured JSON Logging:** FastAPI logs request method, path, response status, latency, and merchant ID.
- **Health Checks:** Set cloud load balancer probes to `/api/v1/health` (10s interval, 2 consecutive failures to mark unhealthy).
- **PostgreSQL Telemetry:** Monitor connection pool utilization (`DB_POOL_SIZE`), slow queries (>200ms), and transaction locks.
- **AI Engine Telemetry:** Track exception investigation completion latency and fallback rates.

---

## 7. Current Deployment Status Note

> **Note:** The current workspace is verified in a local development and evaluation environment running against a local PostgreSQL 18 development instance. Public cloud production deployment requires configuring the target cloud infrastructure (e.g. AWS ECS / GCP Cloud Run + Cloud SQL) following the steps outlined above.
