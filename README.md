# HX-IDL Proj

Hybrid Explainable Intrusion Detection for IoT Botnet Traffic

## IoT Device Intrusion Detection using Hybrid ML

### Overview
This project detects IoT network attacks (Mirai, Gafgyt) using a hybrid ML approach combining:
- XGBoost (supervised learning)
- Autoencoder (anomaly detection)
- Fusion model (final decision layer)

### Dataset
Dataset used: Danmini Doorbell IoT dataset
Stored externally in Google Drive.

### Setup Instructions

#### 1. Clone repository
git clone https://github.com/your-username/IoT-Device-Intrusion-Detection.git
cd IoT-Device-Intrusion-Detection

#### 2. Install dependencies
pip install -r requirements.txt

#### 3. Link dataset
Place dataset in:
data/Danmini_Doorbell/

Or update config path accordingly.

#### 4. Run pipeline
python scripts/run_pipeline.py build --profile quick --run-name final_run

#### 5. Results
Outputs:
- Metrics
- Model artifacts
- Evaluation reports

### Model Performance
- Accuracy: 0.9995
- F1 Score: 0.9997
- AUROC: 1.0

### Notebook
Refer to:
notebooks/final_notebook.ipynb

## 1) Executive Summary

This project implements an end-to-end, research-oriented and production-aware Intrusion Detection System (IDS) pipeline for IoT botnet detection using the N-BaIoT style Doorbell device traffic (benign + Gafgyt + Mirai families).

The core contribution is a hybrid detection stack:
- Baseline supervised multiclass detector (XGBoost)
- SHAP-guided attribution weighting
- Attribution-guided autoencoder (AGA) for reconstruction-based anomaly scoring
- Dual-consistency anomaly scoring (classifier score + reconstruction error)
- Fusion classifier for final attack/benign decision
- LOAO (Leave-One-Attack-Out) evaluation for unseen-attack robustness

The repository also includes:
- Data engineering and validation pipeline
- Experiment tracking hooks (MLflow)
- DVC stage orchestration
- Promotion gates and model versioning
- FastAPI inference service
- Monitoring and drift analysis modules
- High-level automation runner for one-command/two-command operations

## 2) Current Project Status

### Build and implementation status
- Phase-wise implementation is complete (Phase 1 through Phase 7).
- Master automation runner is implemented.
- Core scripts and modules compile clean.

### What remains (important)
- Full runtime test campaign and evaluation reporting should still be executed as a final validation pass in your environment.
- You should run the experiment outputs and collect final numbers/tables for thesis/report submission.

In short: engineering build is complete; final empirical validation and formal testing report generation are the remaining work.

### Completion verdict (plain answer)
- Implementation: complete for the planned pipeline (Phases 1 to 7 + automation).
- Remaining before final sign-off: execution validation, testing, and final metric reporting.
- Net status: build is done; evaluation closure is pending.

## 3) Problem Statement and Academic Context

### Problem
Traditional IDS systems may struggle with:
- Distribution shift in IoT traffic
- Unseen attack variants
- Black-box behavior with limited interpretability

### Project goal
Build a hybrid IDS that improves robustness and interpretability while preserving deployability.

### Research perspective
This project is designed to satisfy both:
- Academic evaluation dimensions (novelty, ablation, robustness, reproducibility)
- Enterprise evaluation dimensions (operationalization, promotion gates, API, monitoring, drift, retraining decisions)

## 4) Novelty Claim

The novelty is not "just another classifier", but a structured hybrid methodology:

1. SHAP-attribution weighting before reconstruction learning
- Features are weighted using SHAP-derived importance, then used by the autoencoder.
- This biases representation learning toward explanatory/decision-relevant dimensions.

2. Dual-consistency anomaly score
- Anomaly evidence is fused from:
  - Discriminative signal: classifier attack probability
  - Generative signal: reconstruction error
- This avoids relying on a single failure mode.

3. Fusion-level final decision model
- A fusion classifier consumes multiple signals (classifier score, reconstruction score, dual score), producing calibrated binary attack decisions.

4. LOAO robustness protocol for unseen attack types
- Each attack label is held out in turn.
- Benign data is split into disjoint train/test subsets per fold to avoid leakage.
- This directly tests generalization to unknown attack categories.

## 5) High-Level Architecture

### Offline pipeline (build path)
1. Data ingestion + schema validation + preprocessing
2. Split generation (train/val/test)
3. Hybrid model training
4. Experiment 1 (baseline vs hybrid/fusion)
5. Experiment 2 (LOAO)
6. Promotion gate evaluation + version manifest
7. Drift analysis and retraining decision artifacts

### Online pipeline (serve path)
1. Load preprocessing + model artifacts
2. Expose prediction endpoints via FastAPI
3. Export Prometheus metrics
4. Support monitoring/health endpoints

## 6) Repository Map (Where, What, Why)

### Top-level orchestration
- dvc.yaml: Defines staged reproducible workflow for key phases.
- params.yaml: Additional runtime parameters for data split/preprocessing.
- requirements.txt: Full dependency lock list.

### Config directory
- config/config.yaml: Core project/data paths, preprocessing, splits, output naming.
- config/model_config.yaml: Hybrid model/training settings, MLflow config, registry thresholds.
- config/inference_config.yaml: API runtime and inference artifact paths.
- config/drift_config.yaml: Data/concept drift thresholds and retraining policy.
- config/monitoring_config.yaml: SLO/threshold definitions for monitoring policy.
- config/automation_config.yaml: Master automation modes/profiles/paths.

### Data layer
- data/schema/data_schema.py: Dataset-aware Pandera schema inference/validation.
- src/data/loader.py: Chunked CSV loading with label/family/source tagging and optional parquet caching.
- src/data/validator.py: Schema validation + quality report generation.
- src/data/splitter.py: Deterministic stratified train/val/test splitter.
- src/data/preprocessor.py: Imputation, clipping, scaling, label encoding, artifact persistence.

### Research core
- src/research/shap_weighting.py: SHAP-based feature weighting utilities.
- src/research/autoencoder.py: Attribution-guided autoencoder implementation.
- src/research/dual_consistency.py: Combined anomaly scoring logic.
- src/research/fusion.py: Fusion classifier abstraction.
- src/research/loao.py: Leave-One-Attack-Out split generator (leakage-safe benign disjoint splits).

### Training and evaluation
- src/training/hybrid_trainer.py: End-to-end hybrid orchestrator (train/eval/artifact save).
- scripts/train_model.py: CLI entrypoint for hybrid training.
- experiments/exp1_ae_vs_baseline.py: Baseline vs hybrid/fusion experiment.
- experiments/exp2_loao.py: LOAO robustness experiment.
- src/evaluation/*: Metrics, evaluator, report and ROC visualization helpers.

### Registry and promotion
- src/registry/model_manager.py: Promotion gating against metric thresholds.
- src/registry/versioning.py: Hash-based version identifiers.
- src/registry/mlflow_registry.py: MLflow logging wrapper.
- scripts/promote_model.py: Promotion decision CLI.

### Inference API
- api/main.py: FastAPI app assembly and startup artifact loading.
- api/routes/predict.py: Single/batch prediction endpoints.
- api/routes/health.py: Health/model info endpoints.
- api/routes/monitoring.py: Lightweight monitoring status endpoint.
- api/schemas/*: Request/response contracts.
- src/inference/model_loader.py: Inference-time artifact loading.
- src/inference/predictor.py: Hybrid prediction logic.
- scripts/run_api.py: API launcher.

### Monitoring and drift
- src/monitoring/*: Performance/health/alerting/metrics helpers.
- src/drift/data_drift.py: PSI + KS drift detection.
- src/drift/concept_drift.py: Performance degradation drift detection.
- src/drift/drift_analyzer.py: Unified drift report generation.
- src/drift/retraining_trigger.py: Retraining trigger policy logic.
- scripts/run_drift_check.py: Drift + retraining decision CLI.
- monitoring/prometheus.yml: Prometheus scrape config.
- monitoring/alerts/alert_rules.yml: Alert rules.
- monitoring/grafana/dashboards/ml_monitoring.json: Dashboard template.

### Pipeline wrappers and automation
- pipelines/data_pipeline.py: Wraps Phase 1 data flow.
- pipelines/training_pipeline.py: Wraps hybrid training.
- pipelines/evaluation_pipeline.py: Runs Exp1 and Exp2.
- pipelines/deployment_pipeline.py: Runs promotion gate and writes summary.
- pipelines/orchestration_pipeline.py: End-to-end phase orchestrator.
- scripts/run_orchestration.py: Entry script for orchestration pipeline.
- scripts/run_pipeline.py: Master automation runner (build / serve / full).

## 7) Data and Labels

### Dataset expectation
Configured default dataset root:
- data/Danmini_Doorbell

Expected sources:
- benign_traffic.csv
- gafgyt_attacks/*.csv
- mirai_attacks/*.csv

### Labeling strategy
During ingestion, each row gets:
- label: benign or attack label (family + file stem)
- attack_family: benign / gafgyt / mirai
- source_file: original CSV filename

## 8) Phase-by-Phase Pipeline (What happens and why)

### Phase 1: Data engineering
- Load CSV data in chunks
- Validate with inferred schema
- Generate quality report
- Perform deterministic stratified split
- Fit preprocessing artifacts on train split
- Save processed splits and preprocessor artifact

Outputs include:
- data/raw/combined_raw.parquet
- data/processed/train.parquet
- data/processed/val.parquet
- data/processed/test.parquet
- data/features/preprocessor.joblib
- data/processed/data_quality_report.json
- data/processed/phase1_stats.json

### Phase 2/3: Hybrid model training and experiments
- Train multiclass XGBoost baseline
- Generate SHAP-based feature weighting
- Train attribution-guided autoencoder on benign-weighted data
- Compute dual-consistency scores
- Train fusion model for binary attack decision
- Run Exp1 and Exp2

Outputs include:
- artifacts/research/models/*
- artifacts/research/reports/phase3_results.json
- artifacts/research/reports/exp1_results.json
- artifacts/research/reports/exp2_loao_results.json

### Phase 4: Registry and promotion
- Read Exp1 + Exp2 results
- Evaluate against configured thresholds
- Generate version hash and promotion manifest

Outputs include:
- artifacts/research/registry/*_manifest.json
- artifacts/research/registry/promotion_decision.json (via CLI output path)

### Phase 5: Inference API
- Load promoted artifacts and preprocessor
- Serve prediction and monitoring routes
- Expose Prometheus metrics

### Phase 6: Monitoring and drift
- Compare reference/current feature distributions (PSI + KS)
- Compare reference/current model behavior (concept drift)
- Assign severity and generate retraining decision

Outputs include:
- artifacts/research/monitoring/drift_report.json
- artifacts/research/monitoring/retraining_decision.json

### Phase 7: End-to-end orchestration
- Run data, training, evaluation, and deployment wrappers sequentially
- Emit orchestration report

Output includes:
- artifacts/research/orchestration/phase7_report.json

## 9) Evaluation Design (Academic)

### Experiment 1: Baseline vs Hybrid/Fusion
Goal:
- Quantify improvement over baseline supervised model.

Primary metrics:
- Baseline weighted F1 (multiclass)
- Hybrid/Fusion F1 (binary)
- Fusion AUROC

### Experiment 2: LOAO robustness
Goal:
- Test generalization to unseen attack labels.

Protocol:
- Hold out one attack type at a time
- Train on benign + remaining attacks
- Test on disjoint benign subset + held-out attack

Primary metrics:
- LOAO mean F1 across held-out attacks
- LOAO minimum F1 (worst-case robustness)

### Recommended thesis/professor reporting package
Include:
- Exp1 metric table (baseline vs hybrid vs fusion)
- Exp2 per-attack LOAO table
- Promotion gate pass/fail table
- Drift scenario analysis (if simulated)
- Error analysis by attack family

## 10) Promotion Gates (Enterprise + Academic Quality Bar)

Configured default thresholds:
- min_fusion_f1: 0.70
- min_fusion_auroc: 0.70
- min_loao_mean_f1: 0.65
- min_loao_min_f1: 0.50
- min_fusion_over_baseline_gap: 0.02

Interpretation:
A model is promotable only if it is both accurate and robust under LOAO, and demonstrates meaningful gain over baseline.

## 11) API Contract

### Base runtime
- Host/port from config/inference_config.yaml (default 0.0.0.0:8000)

### Endpoints
- GET /: service banner and readiness
- GET /metrics: Prometheus metrics
- GET /api/v1/health: health check
- GET /api/v1/model/info: loaded model metadata
- GET /api/v1/monitoring/status: status payload
- POST /api/v1/predict/single: single prediction
- POST /api/v1/predict/batch: batch prediction (max 1024)

### Single prediction request example
{
  "features": {
    "MI_dir_L5_weight": 0.12,
    "MI_dir_L5_mean": 1.45
  }
}

Note: The request must provide all expected feature columns from the fitted preprocessor artifact.

## 12) Monitoring and Drift Policy

### Data drift signals
- PSI per feature
- KS two-sample test per feature

### Concept drift signals
- F1 drop
- AUROC drop
- Confidence shift

### Severity and retraining
- Drift analyzer classifies severity: low / medium / high / critical
- Retraining trigger policy checks severity thresholds
- auto_trigger=false by default, so decision can be advisory unless changed

## 13) Reproducible Runbook

## 13.0) Start Here: Automated Project Execution

If you want to run the project in the intended way, use the automation script below as the primary entrypoint.

Primary script:
- scripts/run_pipeline.py

Modes:
- build: executes orchestration, then drift check (profile-dependent flags)
- serve: launches inference API runtime
- full: runs build first; if successful, starts API runtime

Recommended command flow:
1. Build all offline artifacts
- python scripts/run_pipeline.py build --profile full --run-name final_build

2. Start serving predictions
- python scripts/run_pipeline.py serve

Single-command alternative:
- python scripts/run_pipeline.py full --profile full --run-name final_full

What the automation script triggers internally:
- scripts/run_orchestration.py
- scripts/run_drift_check.py
- scripts/run_api.py

Execution report output:
- artifacts/research/orchestration/automation_report.json

Notes:
- build mode is for data-to-model artifact generation and validation.
- serve mode is a blocking API process.
- full mode is useful for demos, but in operations two-command flow is usually cleaner.

## A. Environment setup (Windows PowerShell)

1. Create/activate virtual environment
2. Install dependencies

Example:
- python -m venv venv
- .\venv\Scripts\Activate.ps1
- pip install -r requirements.txt

## B. DVC-based staged reproduction

- dvc repro phase1_data_pipeline
- dvc repro phase2_hybrid_training
- dvc repro phase3_experiment_exp1
- dvc repro phase3_experiment_exp2
- dvc repro phase4_registry_promotion
- dvc repro phase6_monitoring_drift
- dvc repro phase7_orchestration

Or run all connected stages:
- dvc repro

## C. Script-first execution

1. Train hybrid model
- python scripts/train_model.py --config config/config.yaml --model-config config/model_config.yaml --run-name my_run

2. Run experiments
- python experiments/exp1_ae_vs_baseline.py --config config/config.yaml --model-config config/model_config.yaml
- python experiments/exp2_loao.py --config config/config.yaml --model-config config/model_config.yaml

3. Promotion
- python scripts/promote_model.py --config config/config.yaml --model-config config/model_config.yaml

4. Drift check
- python scripts/run_drift_check.py --config config/drift_config.yaml

5. API serving
- python scripts/run_api.py --config config/inference_config.yaml

## D. Master automation runner

Two-command workflow (recommended):
1. Build pipeline
- python scripts/run_pipeline.py build --profile full --run-name prod_build_01

2. Start API
- python scripts/run_pipeline.py serve

Single-command workflow:
- python scripts/run_pipeline.py full --profile full --run-name prod_full_01

Automation report:
- artifacts/research/orchestration/automation_report.json

## 14) Testing and Evaluation Plan (Remaining Work Checklist)

This section is your final closure checklist.

### Functional testing
- Validate all scripts run in a clean environment
- Verify artifact existence after each phase
- Verify API startup with actual model artifacts
- Smoke test all API endpoints

### ML evaluation testing
- Re-run Exp1 and Exp2 from clean processed data
- Record mean and variance if multiple seeds are used
- Confirm promotion gate behavior with current metrics

### Monitoring/drift testing
- Execute drift check with controlled shifted samples
- Validate severity transitions (low/medium/high/critical)
- Validate retraining decision policy behavior

### Reproducibility testing
- Run pipeline from scratch on fresh clone + fresh venv
- Confirm outputs and reports can be regenerated

### Suggested acceptance criteria
- All required outputs are regenerated without manual file edits
- Promotion decision is deterministic for same inputs/config
- API serves successful predictions with expected latency envelope
- Drift report and retraining decision files are produced

## 15) Key Artifacts to Show Your Professor

Present these first:
- artifacts/research/reports/exp1_results.json
- artifacts/research/reports/exp2_loao_results.json
- artifacts/research/reports/phase3_hybrid_report.md
- artifacts/research/registry/promotion_decision.json
- artifacts/research/orchestration/phase7_report.json
- artifacts/research/orchestration/automation_report.json
- artifacts/research/monitoring/drift_report.json

Why these matter:
- They jointly demonstrate novelty, empirical gains, robustness, operational readiness, and reproducibility.

## 16) Risks, Assumptions, and Limitations

- Some performance targets in monitoring config are policy goals, not guaranteed outcomes without benchmark validation.
- API prediction requests must align with exact trained feature columns.
- The current drift workflow compares selected reference/current datasets and metrics; stronger production setups would include rolling windows and online data contracts.

## 17) Troubleshooting

### Common issue: dependency or import errors
- Ensure active interpreter is the same venv where requirements were installed.

### Common issue: missing artifacts when serving API
- Run build flow first (training + promotion), then start API.

### Common issue: DVC stage fails due path mismatch
- Verify dataset files exist at configured paths in config/config.yaml.

### Common issue: LOAO run count looks incorrect
- Confirm attack label construction from source filenames and family folders.

## 18) Suggested Next Enhancements

- Add formal unit/integration test suite under a dedicated tests directory.
- Add seed sweep and confidence interval reporting for academic rigor.
- Add model calibration plots and threshold optimization study.
- Add containerization + CI workflow for full reproducible deployment.

## 19) Citation/Attribution Notes

If used for academic submission, cite:
- N-BaIoT dataset source
- SHAP method reference
- XGBoost method reference
- Any additional libraries or baseline methods used in your comparative analysis

## 20) GitHub Publishing + Reproducibility Quickstart

This section is the practical checklist for publishing the project and allowing others to recreate your results.

### A. Public repository URL (for report)

Use this line in your final report and replace with your live URL:
- Public GitHub repository URL: `<PASTE_PUBLIC_REPO_URL_HERE>`

Example:
- Public GitHub repository URL: `https://github.com/<username>/<repo-name>`

### B. What is tracked vs ignored in this repository

Data policy used in this repository:
- Tracked under data: only `data/schema/` (source schema definitions)
- Ignored under data: generated/raw/large runtime files

Important:
- If you accidentally commit large data files (`.csv`, `.npy`, etc.), GitHub push will fail due to the 100MB limit.
- Keep data artifacts out of git history unless you intentionally use Git LFS.

### C. First-time push workflow (Windows)

1. Check current remote
- `git remote -v`

2. Ensure branch name
- `git branch -M main`

3. Commit source code and docs
- `git add .`
- `git commit -m "Initial project publication"`

4. Push
- `git push -u origin main`

If push fails with permission error (`403`):
- Sign out/in with the correct GitHub account
- Update `origin` to the correct repository URL if needed

If push fails with large-file error:
- Remove large files from commits/history before pushing
- Re-push with `git push --force-with-lease` only after history cleanup

### D. Recreate project on a fresh machine

1. Clone
- `git clone <PUBLIC_REPO_URL>`
- `cd <repo-folder>`

2. Create environment
- `python -m venv venv`
- `.\venv\Scripts\Activate.ps1`

3. Install dependencies
- `pip install -r requirements.txt`

4. Build artifacts
- `python scripts/run_pipeline.py build --profile full --run-name final_build`

5. (Optional) Serve API
- `python scripts/run_pipeline.py serve`

### E. Minimal reproducibility evidence to include in report

Include paths/output references for:
- `artifacts/research/reports/exp1_results.json`
- `artifacts/research/reports/exp2_loao_results.json`
- `artifacts/research/reports/phase3_hybrid_report.md`
- `artifacts/research/orchestration/automation_report.json`

This demonstrates that another user can reproduce the build and obtain the same report artifacts using documented commands.

---

For day-to-day usage, use scripts/run_pipeline.py as the primary interface. For rigorous ablation and thesis reporting, use experiments/exp1_ae_vs_baseline.py and experiments/exp2_loao.py plus the generated JSON/markdown artifacts.
