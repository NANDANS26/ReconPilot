# ReconPilot — Security Architecture & Data Protection

## 1. Zero Secret Ingestion
- Application configurations adhere to 12-factor principles using `.env` files.
- `.env` and all credential patterns are strictly excluded in `.gitignore`.
- Vite frontend configuration never exposes sensitive credentials (only public `VITE_` prefixed variables).
- Secret keys and authentication tokens (`JWT_SECRET_KEY`, `APP_SECRET_KEY`) reside exclusively in server memory.

## 2. API Security & Input Sanitization
- All HTTP request payloads pass through strict Pydantic schemas.
- Exception handlers suppress internal database connection strings, passwords, and tracebacks from client responses.
- CORS policies whitelist specific origins rather than using wildcard `*` in production.

## 3. Database Security
- Parameterized queries via SQLAlchemy ORM protect against SQL injection vulnerabilities.
- Connection pools enforce connection timeouts to prevent denial-of-service hanging.
- Audit records in PostgreSQL will use append-only permissions for operator logs.

## 4. Production Authentication (Planned: Phase 04)
- JWT access tokens with HMAC-SHA256 signatures.
- Role-based access control (RBAC):
  - `CONTROLLER`: Can view and act on Review Queue, trigger reconciliation runs.
  - `AUDITOR`: Read-only access to historical batches and immutable audit logs.
  - `ADMIN`: System configuration and API key rotation.
