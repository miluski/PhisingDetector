# PhishingDetector

Phishing detection for email and SMS messages. Angular frontend sends message
content to Flask API, which extracts URLs and performs machine-learning
inference in a resource-limited Docker sandbox.

## Features

- Email and SMS analysis
- URL extraction from text, PDF, DOCX and TXT attachments
- Random Forest, XGBoost, MLP, SVM and Logistic Regression models
- Original and extended URL-corpus datasets
- Full and optimized feature modes
- Per-model probabilities, ensemble result and visual explanations
- Resource-limited inference container with controlled network access for
  DNS, TLS and HTTP feature extraction
- HTTPS frontend and API with local self-signed certificate

## Quick start

### Requirements

- Python 3.13
- Docker Engine or Docker Desktop
- Docker Compose v2 (`docker compose`)
- Full project bundle containing ignored `backend/models/` and
  `backend/data/url_corpus/` artifacts

Run from repository root:

```bash
python scripts/setup.py
```

Setup validates Docker and required artifacts, prepares localhost certificate,
builds images, starts services and waits for health checks.

- Frontend: <https://localhost:4000>
- Backend health: <https://localhost:5001/api/health>

Browser displays warning for local self-signed certificate. Accept it only for
`localhost` development environment.

Verify without starting services:

```bash
python scripts/setup.py --check-only
```

View service state and logs:

```bash
docker compose ps
docker compose logs -f backend frontend
```

Stop services:

```bash
docker compose down
```

## Required local artifacts

Models and current URL corpus are distributed with full project bundle but
ignored by Git because of size. Setup requires:

- five model families;
- `full` and `optimized` variants;
- original and `url_corpus` variants;
- scalers for Logistic Regression, MLP and SVM;
- original full/optimized datasets;
- repaired URL-corpus full/optimized training datasets.

Missing files are reported before Docker build with exact relative paths.

## Local development

Docker remains required because backend launches a resource-limited inference
container.

Create environment and install runtime plus development tools:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python scripts/setup.py --prepare-only
```

Start backend from repository root:

```bash
SSL_CERT_FILE="$PWD/certs/localhost/cert.pem" \
SSL_KEY_FILE="$PWD/certs/localhost/key.pem" \
HOST_BACKEND_PATH="$PWD/backend" \
python backend/run.py
```

Start frontend in second terminal:

```bash
cd frontend
npm ci
npm start -- \
  --port 4000 \
  --ssl \
  --ssl-cert ../certs/localhost/cert.pem \
  --ssl-key ../certs/localhost/key.pem
```

Frontend expects API at `https://localhost:5001/api`.

## Dependency files

- `backend/requirements.txt` contains exact-pinned runtime dependency graph.
- `backend/requirements-dev.txt` contains exact-pinned direct development tools
  only and does not repeat runtime packages.
- `frontend/package-lock.json` is canonical frontend dependency lock.

Docker installs runtime manifest only. Local development installs both Python
manifests.

## Project structure

```text
.
├── backend/
│   ├── app/                 Flask resources and composition root
│   ├── config/              Feature, server and path configuration
│   ├── data/                Runtime datasets and research corpora
│   ├── extractors/          Lexical, network and content extraction
│   ├── models/              Runtime and staged ML artifacts
│   ├── notebooks/           Original and URL-corpus experiments
│   ├── scripts/
│   │   ├── data_pipeline/   Corpus ingestion, splitting and features
│   │   ├── maintenance/     Quality, cleanup and duplicate audits
│   │   ├── modeling/        Training, recheck and smoke tests
│   │   └── reporting/       Notebook and report generation
│   ├── services/            Detection, inference and sandbox orchestration
│   ├── tests/               Backend unit and integration tests
│   ├── utils/               Shared research and plotting utilities
│   └── validators/          Request, mode, dataset and model validation
├── docs/                    Data-source and retraining guides
├── frontend/                Angular application
├── results/                 Generated experiment outputs
├── scripts/setup.py         Cross-platform setup entrypoint
└── compose.yaml             Application, sandbox prerequisites and HTTPS
```

## Research commands

Run grouped tools as Python modules from repository root:

```bash
python -m backend.scripts.data_pipeline.check_url_corpus_consistency
python -m backend.scripts.modeling.run_url_corpus_model_smoke_test
python -m backend.scripts.reporting.run_model_notebooks_parallel --help
```

## Repository hygiene

Generated datasets, trained models, experiment results, Python environments,
Node.js dependencies and local certificates are excluded from Git. They may be
required to run the application and must not be treated as disposable cache.

Preview allow-listed obsolete artifacts:

```bash
python -m backend.scripts.maintenance.cleanup_artifacts
```

Delete only reported artifacts:

```bash
python -m backend.scripts.maintenance.cleanup_artifacts --apply
```

Avoid `git clean -X`: it would also remove ignored models, datasets and results.

## API

### Health

```http
GET /api/health
```

### Analyze email

```http
POST /api/analyze-email
Content-Type: multipart/form-data
```

Fields include sender, receiver, content, attachments, models, mode and dataset.

### Analyze SMS

```http
POST /api/analyze-sms
Content-Type: multipart/form-data
```

Fields include content, models, mode and dataset.

### Other endpoints

- `POST /api/extract-features`
- `GET /api/model-accuracies`
- `GET /api/message-examples`

Example health request:

```bash
curl -k https://localhost:5001/api/health
```

## Security model

Each analysis runs in a reusable resource-limited sandbox container with:

- bridge network access required for DNS, TLS and HTTP feature extraction;
- read-only models, data and source mounts;
- 2 GB memory limit;
- 2 CPU limit;
- bounded execution timeout;
- explicit cleanup of orphaned containers.

Backend container requires Docker socket access to create inference sandbox.
Treat Docker socket as privileged host capability. Resource limits and
read-only mounts reduce exposure, but bridge networking is not full network
isolation.

## Tests

Backend:

```bash
python -m pytest backend/tests
python -m backend.scripts.maintenance.check_backend_quality
python -m backend.scripts.maintenance.audit_duplicates
```

Frontend:

```bash
cd frontend
npm ci
npm test -- --watch=false
npm run lint
npm run build
```

Validate Compose:

```bash
docker compose config
```

## Troubleshooting

### Missing project artifacts

Run:

```bash
python scripts/setup.py --check-only
```

Restore reported files from full project bundle. Git clone alone does not
contain large ignored model/corpus artifacts.

### Docker daemon unavailable

Start Docker Desktop or Docker Engine, then verify:

```bash
docker info
docker compose version
```

### Certificate warning

Expected for generated localhost certificate. Do not reuse certificate outside
local development.

### Port conflict

Check processes using ports `4000` and `5001`, or stop earlier Compose stack:

```bash
docker compose down
```

Generated experiment outputs live under `results/`; notebooks remain under
`backend/notebooks/`.
