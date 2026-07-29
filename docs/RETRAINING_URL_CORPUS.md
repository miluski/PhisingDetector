# URL Corpus Retraining Guide

This guide describes the reproducible URL-corpus training workflow. It uses
the active `quality_v4_repaired_*` release, keeps production models untouched,
and writes new artifacts to `backend/models/staged_url_corpus_v4/`.

## 1. Environment

Run all commands from the repository root with the project virtual
environment:

```bash
./.venv/bin/python -m pip install -r backend/requirements.txt
```

Do not train with a different feature policy or a different network mode than
the one used to build the dataset. The active policy is `live + keep`:
unavailable network values remain `-1` and are accompanied by missingness
indicators.

## 2. Active release

The current release is stored in these directories:

- raw, domain-disjoint splits:
  `backend/data/url_corpus/quality_v4_repaired_splits/`;
- extracted feature files:
  `backend/data/url_corpus/quality_v4_repaired_features/`;
- staged models and reports: `backend/models/staged_url_corpus_v4/`;
- selected-feature manifest:
  `backend/config/feature_manifests/url_corpus/quality_v4/selected.json`.

The split sizes are:

| Split | Records | Benign | Phishing |
|---|---:|---:|---:|
| Train | 108,762 | 54,381 | 54,381 |
| Validation | 21,676 | 10,838 | 10,838 |
| Test | 21,000 | 10,500 | 10,500 |

Registered domains are disjoint between the three internal splits. The
independent external test contains 37,790 URLs and must not be used for model
selection.

## 3. Feature modes

The project has three canonical modes:

- `full`: 110 prediction features;
- `optimized`: 76 prediction features defined by the shared fixed subset;
- `selected`: 48 prediction features selected from the training split only.

The `selected` manifest is applied unchanged to validation and test. The old
name `optimized_v2` is accepted only as a backward-compatible request alias;
new datasets, models and reports must use `selected`.

The same `URLFeatureService` and `FeatureConfig` are used for dataset creation,
external recheck and application inference. Do not create a second feature
mapping in a notebook or frontend.

## 4. Build the raw corpus (optional rebuild)

Use public phishing sources and trusted benign roots. Hard negatives should
include legal paths such as `login`, `account`, `secure`, `support`, `payment`,
`verify`, `docs` and `pricing`.

```bash
./.venv/bin/python backend/scripts/build_quality_url_corpus.py \
  --input backend/data/url_corpus/raw_urls_phreshphish.csv \
         backend/data/url_corpus/raw_urls_phreshphish_extra.csv \
         backend/data/url_corpus/raw_urls_phishing_db_tranco.csv \
  --trusted-domains backend/data/url_corpus/benign_roots_tranco.csv \
  --hard-negative-per-domain 15 \
  --minimum-benign-roots 40000 \
  --exclude-benign-root-domains-seen-in-phishing \
  --output backend/data/url_corpus/raw_urls_quality_v4.csv \
  --per-class-limit 150000 --seed 42
```

## 5. Build and repair domain-disjoint splits (optional rebuild)

```bash
./.venv/bin/python backend/scripts/build_url_corpus_splits.py \
  --input backend/data/url_corpus/raw_urls_quality_v4.csv \
  --output-dir backend/data/url_corpus/quality_v4_splits \
  --group-by registered_domain --dedupe template \
  --balance-per-split --min-hard-benign-per-split 1000 --seed 42
```

If a registered-domain collision is found, repair it into a new directory;
never overwrite the source cache:

```bash
./.venv/bin/python backend/scripts/repair_split_domain_overlap.py \
  --raw-dir backend/data/url_corpus/quality_v4_splits \
  --feature-dir backend/data/url_corpus/quality_v4_features \
  --output-raw-dir backend/data/url_corpus/quality_v4_repaired_splits \
  --output-feature-dir backend/data/url_corpus/quality_v4_repaired_features
```

## 6. Extract features

Extract `full` once for each split. Derive `optimized` as a column subset and
select `selected` from the training file; do not run a second network pass for
each feature mode.

```bash
for split in train val test; do
  ./.venv/bin/python backend/scripts/build_url_corpus_dataset.py \
    --input backend/data/url_corpus/quality_v4_repaired_splits/raw_${split}.csv \
    --mode full --network-mode live --missing-policy keep \
    --include-metadata --failure-policy keep-empty \
    --output backend/data/url_corpus/quality_v4_repaired_features/dataset_full_${split}.csv
done

./.venv/bin/python backend/scripts/derive_optimized_features.py \
  --full-dir backend/data/url_corpus/quality_v4_repaired_features \
  --output-dir backend/data/url_corpus/quality_v4_repaired_features

./.venv/bin/python backend/scripts/select_url_corpus_features.py \
  --full-dir backend/data/url_corpus/quality_v4_repaired_features \
  --output-dir backend/data/url_corpus/quality_v4_repaired_features
```

The dataset builder supports resumable temporary files. Keep the cache when a
long extraction is interrupted; do not delete completed `.tmp.done` markers
until the final CSV has been verified.

## 7. Validate before training

Run the check-only mode. It verifies feature schemas, class balance, category
coverage and registered-domain disjointness without fitting a model:

```bash
./.venv/bin/python backend/scripts/retrain_url_corpus_all_models.py \
  --feature-dir backend/data/url_corpus/quality_v4_repaired_features \
  --modes full optimized selected \
  --minimum-category-samples 100 --check-only
```

Expected feature counts are 110, 76 and 48, with the same domain counts in all
three modes.

## 8. Train staged models

Training writes only to the staged directory. Production files under
`backend/models/` are not replaced:

```bash
./.venv/bin/python backend/scripts/retrain_url_corpus_all_models.py \
  --feature-dir backend/data/url_corpus/quality_v4_repaired_features \
  --output-dir backend/models/staged_url_corpus_v4 \
  --modes full optimized selected
```

The script calibrates the decision threshold on validation data. Logistic
regression, SVM and MLP use scalers; Random Forest and XGBoost use unscaled
features. The report contains training time, test metrics and quality gates.

## 9. Recheck staged models

Run the grouped internal test, independent external test and fixed safe/phishing
cases:

```bash
./.venv/bin/python backend/scripts/recheck_url_corpus_models.py \
  --model-root backend/models/staged_url_corpus_v4 \
  --feature-dir backend/data/url_corpus/quality_v4_repaired_features \
  --output-dir results/url_corpus/recheck_v4
```

For a fast schema check, skip live external extraction and fixed cases:

```bash
./.venv/bin/python backend/scripts/recheck_url_corpus_models.py \
  --modes full optimized selected --skip-external --skip-real-cases \
  --output-dir results/url_corpus/recheck_v4
```

The report files are `recheck_metrics.csv`, `recheck_report.json`,
`recheck_summary.md` and `real_url_cases.csv`. The summary is written in
English. A model is not production-ready until the external quality gate and
the fixed-case checks pass.

## 10. Threshold and parameter tuning

Notebook utilities in `backend/utils/threshold_calibration.py` select model
parameters on training data and thresholds on validation data. The preferred
operating point targets FPR no greater than 10% and recall no lower than 80%
when such a point exists. If the constraints conflict, record the selected
trade-off and do not tune against the final test labels.

## 11. External and temporal testing

An independent test should come from a new source or a later collection period.
It must contain `url`, `label` and, for a temporal test, `collected_at`. Never
reuse training domains or choose a threshold on the external labels.

## 12. Archive old artifacts

Preview obsolete generated files:

```bash
./.venv/bin/python backend/scripts/cleanup_project_versions.py
```

Move them only after verification:

```bash
./.venv/bin/python backend/scripts/cleanup_project_versions.py --apply
./.venv/bin/python backend/scripts/cleanup_project_versions.py \
  --apply --include-models
```

Archiving is reversible. Do not use destructive deletion for datasets, models,
reports or caches that may be required for thesis reproducibility.

## 13. Promotion rule

Promote a staged model only after checking:

1. internal test metrics;
2. external metrics and AUC-ROC;
3. benign-root and hard-negative FPR;
4. fixed safe/phishing cases;
5. feature-policy compatibility;
6. resource usage and training time.

Record the promoted artifact, dataset release, feature mode, policy version,
threshold and evaluation date so the result can be reproduced.
