# ============================================================================
# ANTIGRAVITY DEPLOYMENT CONFIGURATION
# Fairness Proxy — Backend Setup Guide
# ============================================================================
# Paste these configurations into Antigravity to deploy the full stack.
# Each section is labelled with the exact Antigravity panel it targets.
# ============================================================================


# ── SECTION 1: DATABASE SCHEMA (Antigravity → Database → SQL Editor) ────────
# Run this SQL to initialise the PostgreSQL schema.
# Antigravity auto-migrates if you paste this under "Schema" → "Raw SQL".

SQL_SCHEMA = """
-- ============================================================
-- Fairness Proxy Database Schema v1.0
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Audit log: one row per request processed by the proxy
CREATE TABLE IF NOT EXISTS audit_logs (
    id                       SERIAL PRIMARY KEY,
    request_id               VARCHAR(64)   NOT NULL UNIQUE,
    timestamp                TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    domain                   VARCHAR(64)   NOT NULL,
    verdict                  VARCHAR(16)   NOT NULL CHECK (verdict IN ('PASS', 'MITIGATE', 'BLOCK')),
    action_taken             VARCHAR(32)   NOT NULL,
    original_score           FLOAT         NOT NULL,
    final_score              FLOAT         NOT NULL,
    original_decision        BOOLEAN       NOT NULL,
    final_decision           BOOLEAN       NOT NULL,
    counterfactual_variance  FLOAT         NOT NULL,
    total_protected_shap     FLOAT         NOT NULL,
    rl_reward                FLOAT         NOT NULL,
    max_score_delta          FLOAT         NOT NULL DEFAULT 0.0,
    xai_report_json          TEXT          NOT NULL,
    fairness_metrics_json    TEXT          NOT NULL,
    rl_decision_json         TEXT          NOT NULL,
    payload_hash             VARCHAR(64)   NOT NULL,
    total_latency_ms         FLOAT
);

-- RL episode replay buffer: training data for the PPO agent
CREATE TABLE IF NOT EXISTS rl_episodes (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id          VARCHAR(64),
    state_json          TEXT        NOT NULL,
    action              VARCHAR(16) NOT NULL,
    reward              FLOAT       NOT NULL,
    cf_variance         FLOAT       NOT NULL,
    protected_shap_sum  FLOAT       NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_audit_logs_domain   ON audit_logs (domain);
CREATE INDEX IF NOT EXISTS idx_audit_logs_verdict  ON audit_logs (verdict);
CREATE INDEX IF NOT EXISTS idx_audit_logs_ts       ON audit_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rl_episodes_req     ON rl_episodes (request_id);

-- View: aggregate bias stats per domain (useful for Grafana dashboard)
CREATE OR REPLACE VIEW domain_fairness_stats AS
SELECT
    domain,
    COUNT(*)                              AS total_requests,
    SUM(CASE WHEN verdict = 'PASS'     THEN 1 ELSE 0 END) AS pass_count,
    SUM(CASE WHEN verdict = 'MITIGATE' THEN 1 ELSE 0 END) AS mitigate_count,
    SUM(CASE WHEN verdict = 'BLOCK'    THEN 1 ELSE 0 END) AS block_count,
    ROUND(AVG(counterfactual_variance)::NUMERIC, 6)  AS avg_cf_variance,
    ROUND(AVG(total_protected_shap)::NUMERIC, 4)     AS avg_protected_shap,
    ROUND(AVG(rl_reward)::NUMERIC, 4)                AS avg_rl_reward,
    ROUND(AVG(total_latency_ms)::NUMERIC, 1)         AS avg_latency_ms
FROM audit_logs
GROUP BY domain;

-- View: recent bias incidents (severity > threshold)
CREATE OR REPLACE VIEW recent_bias_incidents AS
SELECT
    request_id,
    timestamp,
    domain,
    verdict,
    action_taken,
    original_score,
    final_score,
    counterfactual_variance,
    total_protected_shap,
    rl_reward
FROM audit_logs
WHERE verdict IN ('MITIGATE', 'BLOCK')
ORDER BY timestamp DESC
LIMIT 100;
"""


# ── SECTION 2: ENVIRONMENT VARIABLES (Antigravity → Settings → Env Vars) ─────
# Copy-paste the following into your Antigravity environment variable panel.

ENV_VARS = {
    # Application
    "APP_VERSION": "1.0.0",
    "DEBUG": "false",
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "json",

    # Database (Antigravity auto-provisions; replace with connection string)
    "DATABASE_URL": "postgresql+asyncpg://<user>:<pass>@<host>:5432/fairness_proxy",

    # Redis (Antigravity Redis addon)
    "REDIS_URL": "redis://<host>:6379/0",

    # Upstream AI target (the 3rd-party AI you are proxying)
    "TARGET_AI_BASE_URL": "https://api.openai.com/v1",
    "TARGET_AI_API_KEY": "<your_upstream_api_key>",
    "TARGET_AI_TIMEOUT_S": "30.0",

    # Counterfactual twin generation
    "NUM_TWINS": "5",
    "TWIN_SEED": "42",

    # Fairness thresholds
    "CF_VARIANCE_THRESHOLD": "0.05",
    "SHAP_PROTECTED_THRESHOLD": "0.15",
    "SHAP_MITIGATE_LOW": "0.05",
    "SHAP_MITIGATE_HIGH": "0.15",

    # RL agent
    "RL_CHECKPOINT_PATH": "checkpoints/ppo_fairness_agent",
    "RL_ALPHA": "1.0",
    "RL_BETA": "2.0",

    # Zero-shot domain classifier
    "CLASSIFIER_MODEL": "cross-encoder/nli-deberta-v3-small",
}


# ── SECTION 3: API ROUTES (Antigravity → Routes → Import OpenAPI) ────────────
# Paste the following OpenAPI route definitions into Antigravity's route panel.
# Alternatively, point Antigravity to your running server's /openapi.json.

ANTIGRAVITY_ROUTES = [
    {
        "name": "Fairness Proxy — Full Pipeline",
        "method": "POST",
        "path": "/v1/proxy/infer",
        "description": (
            "Full 5-stage fairness pipeline. Intercepts, evaluates, and corrects "
            "third-party AI decisions for bias. Returns verdict + XAI report."
        ),
        "tags": ["Fairness Proxy"],
        "auth": "bearer",
        "body_schema": "ProxyRequest",
        "response_schema": "ProxyResponse",
    },
    {
        "name": "Domain Classification Only",
        "method": "POST",
        "path": "/v1/proxy/classify",
        "description": "Dry-run. Detects domain and splits features into merit vs protected.",
        "tags": ["Fairness Proxy"],
    },
    {
        "name": "Generate Counterfactual Twins",
        "method": "POST",
        "path": "/v1/proxy/twins",
        "description": "Dry-run. Generates N counterfactual permutations of the input.",
        "tags": ["Fairness Proxy"],
    },
    {
        "name": "List Audit Logs",
        "method": "GET",
        "path": "/v1/audit/logs",
        "description": "Paginated list of all intercepted decisions with fairness verdicts.",
        "tags": ["Audit"],
    },
    {
        "name": "Get Audit Log Detail",
        "method": "GET",
        "path": "/v1/audit/logs/{request_id}",
        "description": "Full audit log including XAI report and RL decision.",
        "tags": ["Audit"],
    },
    {
        "name": "Aggregate Fairness Statistics",
        "method": "GET",
        "path": "/v1/audit/stats",
        "description": "Aggregate PASS/MITIGATE/BLOCK rates, avg L_CF, avg SHAP.",
        "tags": ["Audit"],
    },
    {
        "name": "RL Episode Replay Buffer",
        "method": "GET",
        "path": "/v1/audit/rl-episodes",
        "description": "Raw RL training episodes for offline PPO fine-tuning.",
        "tags": ["Audit"],
    },
    {
        "name": "Health Check",
        "method": "GET",
        "path": "/health",
        "description": "Liveness probe.",
        "tags": ["Health"],
    },
    {
        "name": "Readiness Check",
        "method": "GET",
        "path": "/health/ready",
        "description": "Readiness probe for Kubernetes/ECS.",
        "tags": ["Health"],
    },
    {
        "name": "Prometheus Metrics",
        "method": "GET",
        "path": "/metrics",
        "description": "Prometheus-format metrics for Grafana scraping.",
        "tags": ["Observability"],
    },
]


# ── SECTION 4: ANTIGRAVITY PROMPT TEMPLATES ──────────────────────────────────
# Paste these as "Prompt Templates" in Antigravity's LLM connector panel.
# They instruct the local LLM assistant on how to generate test payloads
# and interpret XAI reports.

PROMPT_TEMPLATES = {

    "domain_classifier_system": """
You are a domain classification engine for a fairness proxy middleware.
Your task is to determine the decision-making domain of an AI request payload.

Available domains:
- loan_approval: Mortgage, personal loan, business loan decisions
- job_application: Hiring, recruitment, promotion decisions
- healthcare_triage: Medical triage, treatment prioritisation, diagnosis
- insurance_underwriting: Life, health, property, auto insurance
- credit_scoring: Credit card, BNPL, credit limit decisions
- rental_application: Residential rental, property lease decisions

Given a JSON payload, respond ONLY with a JSON object:
{
  "domain": "<domain_name>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<one sentence>"
}

Do not include any other text.
""",

    "xai_report_interpreter": """
You are an AI fairness analyst. Given a SHAP-based XAI report from a fairness proxy,
produce a clear, regulatory-grade explanation of the bias findings.

Your explanation must:
1. State which protected attributes drove the AI decision most strongly
2. Quantify the counterfactual variance (L_CF) in plain English
3. Explain what the RL verdict (PASS/MITIGATE/BLOCK) means for the applicant
4. Suggest what the operator should audit next

Respond in structured paragraphs. Be precise but avoid jargon where possible.
Reference specific SHAP values from the report.
""",

    "test_payload_generator": """
You are a test data generator for an AI fairness proxy.
Generate a realistic JSON payload for the domain: {domain}

Requirements:
- Include 4-6 merit features (income, credit_score, etc.)
- Include 2-4 protected features (age, gender, race, etc.)
- Values must be realistic and internally consistent
- Introduce a SUBTLE bias signal: make the applicant demographically
  disadvantaged but financially qualified (to test the proxy's detection)

Respond ONLY with valid JSON. No comments, no markdown.
""",

    "mitigation_explainer": """
You are a fairness compliance officer explaining an AI decision correction.

Context:
- Original AI score: {original_score}
- Corrected AI score: {corrected_score}
- Protected features that showed bias: {protected_features}
- SHAP values: {shap_values}
- RL action taken: {action}

Write a 3-paragraph explanation suitable for:
1. The applicant (plain language, rights-focused)
2. The lending/hiring officer (operational impact)
3. The regulatory auditor (technical compliance language)
""",
}


# ── SECTION 5: PROMETHEUS SCRAPE CONFIG (configs/prometheus.yml) ─────────────

PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fairness_proxy'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
"""


# ── SECTION 6: STARTUP SEQUENCE ──────────────────────────────────────────────

STARTUP_SEQUENCE = """
Step-by-step deployment checklist for Antigravity:

1. PROVISION DATABASE
   → Antigravity: Database → New PostgreSQL instance → name: fairness_proxy
   → Run SQL from SECTION 1 above in the SQL editor
   → Copy the connection string into DATABASE_URL env var

2. PROVISION REDIS
   → Antigravity: Add-ons → Redis → copy URL into REDIS_URL

3. SET ENVIRONMENT VARIABLES
   → Paste all key-value pairs from SECTION 2 into Settings → Environment

4. DEPLOY API
   → Antigravity: New Service → Docker → point to your registry
   → Or: Antigravity: New Service → Git → repo root → Dockerfile detected automatically
   → Port: 8000
   → Health check path: /health/ready

5. IMPORT API ROUTES
   → Antigravity: API Gateway → Import → OpenAPI URL: https://<your_domain>/openapi.json
   → Or manually add routes from SECTION 3

6. CONFIGURE LLM CONNECTOR
   → Antigravity: AI → LLM Connector → paste prompt templates from SECTION 4
   → Model: gpt-4o-mini (or local Ollama: mistral)
   → Temperature: 0.1 (deterministic for classification)

7. TRAIN RL AGENT (first time)
   → Run: python scripts/train_rl_agent.py --iterations 200 --synthetic 10000
   → Checkpoint saves to: checkpoints/ppo_fairness_agent
   → Mount checkpoints/ as a persistent volume in Antigravity

8. SET UP PROMETHEUS + GRAFANA
   → Antigravity: Observability → Prometheus → paste config from SECTION 5
   → Grafana: Import dashboard ID 12900 (FastAPI template) + add custom panels for:
       * Verdict distribution (PASS/MITIGATE/BLOCK) over time
       * Average L_CF per domain
       * Average protected SHAP per domain
       * RL reward trend (should trend toward 0 as agent improves)

9. VERIFY DEPLOYMENT
   curl -X POST https://<your_domain>/v1/proxy/infer?mock=true \\
     -H 'Content-Type: application/json' \\
     -d '{
       "target_endpoint": "/v1/decisions/loan",
       "domain": "loan_approval",
       "payload": {
         "income": 75000,
         "credit_score": 720,
         "loan_amount": 300000,
         "employment_years": 6,
         "age": 52,
         "gender": "female",
         "race": "Black"
       }
     }'

   Expected response: verdict=MITIGATE or BLOCK (mock upstream is intentionally biased)
"""

if __name__ == "__main__":
    print("Antigravity configuration loaded. See each SECTION above.")
    print(f"Total routes: {len(ANTIGRAVITY_ROUTES)}")
    print(f"Total env vars: {len(ENV_VARS)}")
    print(f"Total prompt templates: {len(PROMPT_TEMPLATES)}")
