# REPOSITORY_CLEANUP_AUDIT.md
## ASTRA — Repository Inventory & Cleanup Audit Plan

---

### 1. Safety & Version Control Baseline
- **Backup Branch:** `cleanup-pre-judge-backup`
- **Baseline Git Commit Hash:** `62d236a783e6e1131e149317bc7961a8ff8d77df`
- **Safety Policy:**
  - Zero modifications to working algorithms, model weights, or scientific logic.
  - All historical root development notes consolidated into structured `docs/` and `docs/archive/`.
  - Production model checkpoints verified with SHA-256 and mapped in `models/model_manifest.json`.

---

### 2. Full Repository Inventory & Proposed Action Matrix

| Path | Type | Purpose | Referenced By | Runtime? | Train/Test? | Current? | Recommended Action | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`README.md`** | DOC | Main Project Overview | Git / Users / Judges | Yes | No | Needs Update | **`KEEP & UPGRADE`** | Rewrite to provide a comprehensive 5-minute judge overview. |
| **`SETUP_AND_RUN.md`** | DOC | Setup instructions | Users | Yes | No | Yes | **`MOVE`** $\rightarrow$ `RUNNING_ASTRA.md` | Standardize top-level execution guide. |
| **`ARCHITECTURE.md`** | DOC | High-level system architecture | Judges | Yes | No | New | **`CREATE`** | Official 14-stage architecture document. |
| **`TECH_STACK.md`** | DOC | Technologies used | Judges | Yes | No | New | **`CREATE`** | Complete justification for Python, PyTorch, SciPy, CUDA, PyQt6. |
| **`RESULTS.md`** | DOC | Authoritative benchmark results | Judges | Yes | No | New | **`CREATE`** | Consolidates latest 1,050-capture verified metrics. |
| **`KNOWN_LIMITATIONS.md`** | DOC | Physical/mathematical limits | Judges | Yes | No | New | **`CREATE`** | Documents dimensionless raw IQ Fs and unconstrained PR bounds. |
| **`FUTURE_WORK.md`** | DOC | Roadmap & extensions | Judges | Yes | No | New | **`CREATE`** | Realistic hardware/live SDR deployment roadmap. |
| **`STAGE7_TO_10_INTEGRATION_AUDIT.md`** | DOC | Development notes | Internal | No | No | Legacy | **`MOVE`** $\rightarrow$ `docs/archive/` | Historical development audit artifact. |
| **`MASTER_BENCHMARK_RECONCILIATION_AUDIT.md`** | DOC | Reconciliation notes | Internal | No | No | Legacy | **`MOVE`** $\rightarrow$ `docs/archive/` | Historical reconciliation record. |
| **`MASTER_BENCHMARK_RECONCILED_RESULTS.md`** | DOC | Benchmark run notes | Internal | No | No | Legacy | **`MOVE`** $\rightarrow$ `docs/archive/` | Replaced by root `RESULTS.md`. |
| **`ASTRA_FINAL_END_TO_END_BENCHMARK_REPORT.md`** | DOC | Master benchmark report | Internal | No | No | Legacy | **`MOVE`** $\rightarrow$ `docs/archive/` | Replaced by root `RESULTS.md`. |
| **`FIRST_FAILURE_STAGE_ANALYSIS.csv`** | DATA | Stage failure log | Internal | No | No | Legacy | **`MOVE`** $\rightarrow$ `docs/archive/` | Historical analysis artifact. |
| **`OFFICIAL_REQUIREMENT_TRACEABILITY_DRAFT.md`** | DOC | Requirement draft | Internal | No | No | Current | **`MOVE`** $\rightarrow$ `docs/REQUIREMENTS.md` | Formalize as canonical requirements documentation. |
| **`REQUIREMENT_TRACEABILITY_MATRIX.csv`** | DATA | Requirement CSV | Judges | Yes | No | Current | **`MOVE`** $\rightarrow$ `docs/REQUIREMENT_TRACEABILITY_MATRIX.csv` | Move to clean `docs/` directory. |
| **`OFFICIAL_REQUIREMENT_COMPLIANCE_REPORT.md`** | DOC | Formal compliance report | Judges | Yes | No | Current | **`MOVE`** $\rightarrow$ `docs/COMPLIANCE_REPORT.md` | Standardize documentation hierarchy. |
| **`checkpoints/`** | DIR | Production model weights | Production code | Yes | Yes | Yes | **`KEEP`** | Contains active `astra_resnet1d_v2.pt`, `astra_spectrogram_cnn_v2.pt`, `symbol_rate_ranker.joblib`. |
| **`best_model_resnet1d.pt`** | FILE | Root redundant weight (18.9 MB) | CLI / Doctor fallback | No | No | Legacy | **`MOVE`** $\rightarrow$ `models/optional/` | Prevent root clutter. |
| **`ASTRA_Modulation_Model_Starter/`** | DIR | Legacy training sandbox (83.6 MB) | Historical | No | Yes | Legacy | **`REVIEW & ARCHIVE`** | Keep in `docs/archive/` or separate folder without polluting active `src/`. |
| **`legacy_modulation_v1/`** | DIR | Deprecated V1 models (68.2 MB) | None active | No | No | Legacy | **`REVIEW & ARCHIVE`** | Deprecated baseline models. |
| **`outputs/` & `output/`** | DIR | Generated test caches (38 MB) | Test runs | No | No | Temporary | **`CLEAN / GITIGNORE`** | Ignore test artifact caches. |
| **`datasets/ASTRA_FINAL_TEST_SET/`** | DATA | Official test captures | Benchmark harness | Yes | Yes | Yes | **`KEEP`** | Official benchmark validation suite. |
| **`tests/`** | DIR | Test suites | Pytest | Yes | Yes | Yes | **`KEEP & ORGANIZE`** | Unit, integration, and smoke test suites. |
| **`scripts/`** | DIR | Utility and demo scripts | Users | Yes | Yes | Yes | **`CLEAN & RENAME`** | Add `run_demo.py` and `validate_installation.py`. |
| **`astra_gui/`** | SRC | Scientific PyQt6 GUI | Desktop users | Yes | No | Yes | **`KEEP`** | Production scientific desktop interface. |
| **`astra_*/`** | SRC | Pipeline algorithm packages | Pipeline | Yes | Yes | Yes | **`KEEP`** | Core production engines. |
| **`__pycache__/`, `.pytest_cache/`, `*.pyc`** | CACHE | Python/pytest bytecode | System | No | No | Ephemeral | **`DELETE / IGNORE`** | Clean development clutter. |
