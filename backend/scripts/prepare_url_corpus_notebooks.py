from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.normalize_notebook_labels import normalize_source
except ModuleNotFoundError:
    from normalize_notebook_labels import normalize_source


SOURCE_NOTEBOOKS = (
    "0_comprehensive_statistical_analysis.ipynb",
    "1_logistic_regression_phishing_detection.ipynb",
    "2_random_forest_phishing_detection.ipynb",
    "3_xgboost_phishing_detection.ipynb",
    "4_svm_phishing_detection.ipynb",
    "5_mlp_phishing_detection.ipynb",
    "6_comprehensive_model_testing.ipynb",
    "7_dataset_imbalance_research.ipynb",
)

URL_UNUSED_MODEL_SELECTION_IMPORTS = {
    "1_logistic_regression_phishing_detection.ipynb": {"train_test_split"},
    "2_random_forest_phishing_detection.ipynb": {"train_test_split"},
    "3_xgboost_phishing_detection.ipynb": {"train_test_split"},
    "5_mlp_phishing_detection.ipynb": {"train_test_split"},
}

# Deliberate post-generation corrections that must survive regeneration.
CANONICAL_CORRECTED_CELL_IDS = {
    "0266d804", "e084eb18", "ba869cfd", "38d465be", "2581765b", "bc09f0e9",
    "7b30fa83", "02a03191", "4b0ee14a", "a94e9c72", "4fdc997b", "cfc8ce35", "2a5fbffc",
    "0ea77dd7", "d16ed136", "b033dd56", "42d26bcc", "e19b8316", "3ca25c5d",
    "563fd230", "e9598cc2", "eb89d285", "3e8b6741", "1beefb18", "aa953871", "6f087e72",
    "fea02c85", "288128e0", "a60b5168", "32d48eb0", "6dd157ef",
    "b7d61617",
}


BOOTSTRAP_CELL = {
    "cell_type": "code",
    "metadata": {
        "language": "python"
    },
    "source": [
        "from pathlib import Path\n",
        "import os\n",
        "import sys\n",
        "\n",
        "start_dir = Path.cwd().resolve()\n",
        "backend_root = None\n",
        "for candidate in (start_dir, *start_dir.parents):\n",
        "    if (candidate / 'utils').is_dir() and (candidate / 'notebooks').is_dir():\n",
        "        backend_root = candidate\n",
        "        break\n",
        "    nested_backend = candidate / 'backend'\n",
        "    if (nested_backend / 'utils').is_dir() and (nested_backend / 'notebooks').is_dir():\n",
        "        backend_root = nested_backend\n",
        "        break\n",
        "\n",
        "if backend_root is None:\n",
        "    raise RuntimeError(\n",
        "        f'Cannot locate backend root from Jupyter working directory: {start_dir}'\n",
        "    )\n",
        "\n",
        "notebooks_dir = backend_root / 'notebooks'\n",
        "os.chdir(notebooks_dir)\n",
        "if str(backend_root) not in sys.path:\n",
        "    sys.path.insert(0, str(backend_root))\n",
        "\n",
        "print(f'Notebook CWD: {Path.cwd()}')\n",
        "print(f'Backend root in sys.path: {backend_root}')\n"
    ],
    "outputs": [],
    "execution_count": None
}


MISSING_MINUS_ONE_ALWAYS_CELL = {
    "cell_type": "code",
    "metadata": {
        "language": "python"
    },
    "source": [
        "\n",
        "def generate_missing_minus_one_artifacts(df, top_features, html_dir, png_dir):\n",
        "    missing_rows = []\n",
        "    for feature in top_features:\n",
        "        data0_all = df[df['phishing'] == 0][feature]\n",
        "        data1_all = df[df['phishing'] == 1][feature]\n",
        "        excluded0 = int((data0_all == -1).sum())\n",
        "        excluded1 = int((data1_all == -1).sum())\n",
        "\n",
        "        missing_rows.append({\n",
        "            'Cecha': get_readable_name(feature),\n",
        "            'Bez phishingu': (excluded0 / len(data0_all)) * 100 if len(data0_all) else 0.0,\n",
        "            'Phishing': (excluded1 / len(data1_all)) * 100 if len(data1_all) else 0.0,\n",
        "            'Bez phishingu (count)': excluded0,\n",
        "            'Phishing (count)': excluded1,\n",
        "        })\n",
        "\n",
        "    missing_df = pd.DataFrame(missing_rows)\n",
        "\n",
        "    fig_missing = go.Figure()\n",
        "    fig_missing.add_trace(go.Bar(\n",
        "        name='Bez phishingu',\n",
        "        x=missing_df['Cecha'],\n",
        "        y=missing_df['Bez phishingu'],\n",
        "        marker_color='#636EFA',\n",
        "        hovertemplate=(\n",
        "            '<b>%{x}</b><br>Nie występuje w URL: %{y:.1f}%<br>'\n",
        "            'Liczba: %{customdata}<extra></extra>'\n",
        "        ),\n",
        "        customdata=missing_df['Bez phishingu (count)']\n",
        "    ))\n",
        "    fig_missing.add_trace(go.Bar(\n",
        "        name='Phishing',\n",
        "        x=missing_df['Cecha'],\n",
        "        y=missing_df['Phishing'],\n",
        "        marker_color='#EF553B',\n",
        "        hovertemplate=(\n",
        "            '<b>%{x}</b><br>Nie występuje w URL: %{y:.1f}%<br>'\n",
        "            'Liczba: %{customdata}<extra></extra>'\n",
        "        ),\n",
        "        customdata=missing_df['Phishing (count)']\n",
        "    ))\n",
        "\n",
        "    y_max = float(max(missing_df['Bez phishingu'].max(), missing_df['Phishing'].max()))\n",
        "    if y_max <= 0:\n",
        "        y_max = 1.0\n",
        "\n",
        "    fig_missing.update_layout(\n",
        "        title='Procent Przypadków gdzie Cecha Nie Występuje w URL (wartości -1)',\n",
        "        title_font=dict(size=18, color='black', family='Arial Black'),\n",
        "        xaxis_title='Cecha',\n",
        "        yaxis_title='Procent przypadków z wartością -1 [%]',\n",
        "        xaxis_title_font=dict(size=14, color='black', family='Arial Black'),\n",
        "        yaxis_title_font=dict(size=14, color='black', family='Arial Black'),\n",
        "        barmode='group',\n",
        "        height=500,\n",
        "        width=1400,\n",
        "        showlegend=True,\n",
        "        legend_title_text='Klasa',\n",
        "        legend=dict(font=dict(size=12)),\n",
        "        xaxis=dict(\n",
        "            tickangle=-45,\n",
        "            tickfont=dict(size=11, color='black'),\n",
        "            showgrid=True,\n",
        "            gridwidth=1,\n",
        "            gridcolor='lightgray',\n",
        "            showline=True,\n",
        "            linewidth=2,\n",
        "            linecolor='black'\n",
        "        ),\n",
        "        yaxis=dict(\n",
        "            tickfont=dict(size=12, color='black'),\n",
        "            showgrid=True,\n",
        "            gridwidth=1,\n",
        "            gridcolor='lightgray',\n",
        "            showline=True,\n",
        "            linewidth=2,\n",
        "            linecolor='black',\n",
        "            range=[0, y_max * 1.15]\n",
        "        ),\n",
        "        plot_bgcolor='white',\n",
        "        bargap=0.2,\n",
        "        bargroupgap=0.1\n",
        "    )\n",
        "\n",
        "    html_path = f'{html_dir}/missing_values_minus1.html'\n",
        "    png_path = f'{png_dir}/missing_values_minus1.png'\n",
        "    fig_missing.write_html(html_path)\n",
        "    fig_missing.write_image(png_path, scale=2, width=1400, height=500)\n",
        "    print(f'✓ Wymuszony zapis wykresu -1: {png_path}')\n",
        "    print(f'✓ Wymuszony zapis HTML -1: {html_path}')\n",
        "\n",
        "\n",
        "top_features_for_missing = feature_importance['Cecha'].head(8).tolist()\n",
        "generate_missing_minus_one_artifacts(df_clean, top_features_for_missing, html_dir, png_dir)\n"
    ],
    "outputs": [],
    "execution_count": None
}


SPLIT_AWARE_LOAD_CELL = """from utils.split_dataset_loader import (
    build_train_test_data,
    feature_columns,
    load_split_frames,
    print_split_summary,
)

QUALITY_SPLIT_DIR = backend_root / 'data' / 'url_corpus' / 'quality_v4_repaired_features'
split_frames = {
    'Pełny zbiór': load_split_frames(QUALITY_SPLIT_DIR, 'dataset_full'),
    'Zoptymalizowany zbiór': load_split_frames(QUALITY_SPLIT_DIR, 'dataset_optimized'),
}
print_split_summary(split_frames)
datasets = split_frames
"""


SPLIT_AWARE_DATA_SPLITS_CELL = """data_splits = {}

for name, frames in datasets.items():
    train_df = frames['train']
    model_feature_columns = feature_columns(train_df)
    train_class_counts = train_df['phishing'].value_counts()
    positive_count = int(train_class_counts.get(1, 0))
    if positive_count == 0:
        raise ValueError(f'Brak klasy phishing w treningowym splicie: {name}')
    scale_pos_weight = float(train_class_counts.get(0, 0) / positive_count)

    data_splits[name] = {
        'frames': frames,
        'X': train_df[model_feature_columns],
        'y': train_df['phishing'],
        'feature_columns': model_feature_columns,
        'feature_count': len(model_feature_columns),
        'scale_pos_weight': scale_pos_weight,
    }

    print(f"\\n{name}:")
    print(f"  • Liczba cech: {len(model_feature_columns)}")
    for split_name, df in frames.items():
        counts = df['phishing'].value_counts()
        print(f"  • {split_name}: {len(df):,} próbek")
        print(f"    - Brak phishingu: {counts.get(0, 0):,}")
        print(f"    - Phishing: {counts.get(1, 0):,}")
"""


SPLIT_AWARE_TRAIN_TEST_CELL = """from sklearn.preprocessing import StandardScaler

train_test_data = build_train_test_data(datasets)

for name, data in train_test_data.items():
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    data_splits[name]['X'] = X_train
    data_splits[name]['y'] = y_train
    data_splits[name]['X_train'] = X_train
    data_splits[name]['X_val'] = X_val
    data_splits[name]['X_test'] = X_test
    data_splits[name]['y_train'] = y_train
    data_splits[name]['y_val'] = y_val
    data_splits[name]['y_test'] = y_test
    data_splits[name]['X_train_scaled'] = X_train_scaled
    data_splits[name]['X_val_scaled'] = X_val_scaled
    data_splits[name]['X_test_scaled'] = X_test_scaled
    data_splits[name]['scaler'] = scaler

    print(f"\\n{name}:")
    print(f"  • Zbiór uczący: {X_train.shape[0]:,} próbek")
    print(f"  • Zbiór do testowania podczas uczenia: {X_val.shape[0]:,} próbek")
    print(f"  • Końcowy zbiór testowy: {X_test.shape[0]:,} próbek")
    print("  • Podział pochodzi z plików quality_v4_repaired_features; train_test_split nie jest używany.")

    train_counts = y_train.value_counts()
    print(f"  • Rozkład klas (uczący):")
    print(f"    - Brak phishingu: {train_counts.get(0, 0):,}")
    print(f"    - Phishing: {train_counts.get(1, 0):,}")
"""


PHASE_MODEL_LOAD_CELL = """model_configs = [
    {
        'name': 'Regresja Logistyczna',
        'dir': 'logistic_regression',
        'pattern': 'logistic_regression_{}_url_corpus_corrected.pkl',
    },
    {
        'name': 'Random Forest',
        'dir': 'random_forest',
        'pattern': 'random_forest_{}_url_corpus_corrected.pkl',
    },
    {
        'name': 'XGBoost',
        'dir': 'xgboost',
        'pattern': 'xgboost_model_{}_url_corpus_corrected.pkl',
    },
    {
        'name': 'SVM',
        'dir': 'svm',
        'pattern': 'svm_model_{}_url_corpus_corrected.pkl',
    },
    {
        'name': 'MLP',
        'dir': 'mlp',
        'pattern': 'mlp_model_{}_url_corpus_corrected.pkl',
    },
]

loaded_models = {}
artifact_modes = {'full': 'full', 'optimized': 'selected'}

print('='*80)
print('ŁADOWANIE MODELI QUALITY V4')
print('='*80)

for config in model_configs:
    model_name = config['name']
    loaded_models[model_name] = {}
    for dataset_type, artifact_mode in artifact_modes.items():
        model_path = (
            MODELS_DIR / config['dir'] /
            config['pattern'].format(artifact_mode)
        )
        if not model_path.exists():
            print(f'❌ {model_name} ({dataset_type}): brak {model_path.name}')
            continue
        with model_path.open('rb') as file:
            payload = pickle.load(file)
        loaded_models[model_name][dataset_type] = payload
        feature_count = len(payload['feature_names'])
        print(f'✅ {model_name} ({dataset_type}): {feature_count} cech')

print(f'\\n✅ Załadowano {sum(len(v) for v in loaded_models.values())} modeli')
"""


PHASE_PREDICTION_HELPERS_CELL = """def prepare_test_data(df, feature_names):
    missing = sorted(set(feature_names).difference(df.columns))
    if missing:
        raise ValueError(f'Brak cech wymaganych przez model: {missing}')
    return df[feature_names], df['phishing']


def positive_probabilities(model, values):
    if hasattr(model, 'predict_proba'):
        return np.asarray(model.predict_proba(values)[:, 1]).reshape(-1)
    return np.asarray(model.predict(values, verbose=0)).reshape(-1)


def predict_with_model(payload, X):
    values = X
    if payload.get('scaler') is not None:
        values = payload['scaler'].transform(values)
    probabilities = positive_probabilities(payload['model'], values)
    threshold = float(payload.get('decision_threshold', 0.5))
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'false_positive_rate': false_positive_rate,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def analyze_misclassifications(X, y_true, y_pred, y_proba, indices):
    misclassified = []
    for idx in indices:
        if y_true.iloc[idx] == y_pred[idx]:
            continue
        misclassified.append({
            'index': idx,
            'true_label': y_true.iloc[idx],
            'predicted_label': y_pred[idx],
            'probability': y_proba[idx],
            'features': X.iloc[idx].to_dict(),
        })
    return misclassified


print('✅ Funkcje pomocnicze zdefiniowane')
"""


LOGISTIC_PAYLOAD_SAVE_CELL = """for dataset_name, model_data in trained_models.items():
    model = model_data['model']
    scaler = train_test_data[dataset_name]['scaler']
    file_suffix = 'full' if 'Pełny' in dataset_name else 'optimized'
    model_filename = f'{MODELS_DIR}/logistic_regression_{file_suffix}_url_corpus.pkl'

    model_payload = {
        'model': model,
        'scaler': scaler,
        'feature_names': train_test_data[dataset_name]['X_train'].columns.tolist(),
        'best_params': model_data['params'],
        'decision_threshold': evaluation_results[dataset_name]['decision_threshold'],
        'threshold_calibration': evaluation_results[dataset_name]['threshold_calibration'],
        'metrics': {
            'accuracy': evaluation_results[dataset_name]['accuracy'],
            'precision': evaluation_results[dataset_name]['precision'],
            'recall': evaluation_results[dataset_name]['recall'],
            'false_positive_rate': evaluation_results[dataset_name]['false_positive_rate'],
            'f1': evaluation_results[dataset_name]['f1'],
            'auc_roc': evaluation_results[dataset_name]['auc_roc'],
            'cv_auc_mean': evaluation_results[dataset_name]['cv_mean'],
            'cv_auc_std': evaluation_results[dataset_name]['cv_std'],
        },
    }
    with open(model_filename, 'wb') as file:
        pickle.dump(model_payload, file)

    print(f"✅ {dataset_name}:")
    print(f"   Model i skalibrowany próg zapisane: {model_filename}")
    print(f"   Parametry: {model_data['params']}")
    print(f"   Próg decyzyjny: {model_payload['decision_threshold']:.4f}")
    print(f"   Test ROC-AUC: {evaluation_results[dataset_name]['auc_roc']:.4f}")
    print()
"""


def _rewrite_text(text: str) -> str:
    """Rewrite dataset paths, corpus labels and model artifact names."""
    updated = _rewrite_bootstrap(text)
    for old, new in _dataset_replacements():
        updated = updated.replace(old, new)
    return normalize_source(_rewrite_artifact_names(updated))


def _rewrite_bootstrap(text):
    """Rewrite corpus name and notebook import bootstrap."""
    bootstrap = (
        "cwd = Path.cwd().resolve(); "
        "os.chdir(cwd.parent) if cwd.name == 'url_corpus' and cwd.parent.name == 'notebooks' else None; "
        "nb_dir = Path.cwd().resolve(); "
        "backend_root = nb_dir.parent if nb_dir.name == 'notebooks' else nb_dir; "
        "sys.path.insert(0, str(backend_root)) if str(backend_root) not in sys.path else None"
    )
    return text.replace("CORPUS_NAME = 'original'", "CORPUS_NAME = 'url_corpus'").replace(
        "sys.path.append('..')", bootstrap
    )


def _dataset_replacements():
    """Return deterministic source-to-URL-corpus path replacements."""
    return (
        ("quality_large_splits", "quality_v4_repaired_features"),
        ("../data/dataset_full_balanced.csv",
         "../data/url_corpus/quality_v4_repaired_features/dataset_full_train.csv"),
        ("../data/dataset_optimized_balanced.csv",
         "../data/url_corpus/quality_v4_repaired_features/dataset_optimized_train.csv"),
        ("../data/dataset_full.csv", "../data/url_corpus/quality_v4_repaired_features/dataset_full_train.csv"),
        ("../data/dataset_optimized.csv",
         "../data/url_corpus/quality_v4_repaired_features/dataset_optimized_train.csv"),
        (
            "backend_root.parent / 'results' / 'original'",
            "backend_root.parent / 'results' / 'url_corpus'",
        ),
        ("backend_root / 'data' / 'test'", "backend_root / 'data' / 'url_corpus' / 'quality_v4_repaired_features'"),
        ("test_dataset_full.csv", "dataset_full_test.csv"),
        ("test_dataset_optimized.csv", "dataset_optimized_test.csv"),
        ("../data/test", "../data/url_corpus/test"),
    )


def _rewrite_artifact_names(text):
    """Append URL-corpus suffixes to legacy model filenames."""
    for old, new in (("_{}.pkl", "_{}_url_corpus.pkl"), ("_{dataset_type}.pkl", "_{dataset_type}_url_corpus.pkl"),
                     ("_{file_suffix}.pkl", "_{file_suffix}_url_corpus.pkl"), ("_full.pkl", "_full_url_corpus.pkl"),
                     ("_optimized.pkl", "_optimized_url_corpus.pkl")):
        text = text.replace(old, new)
    return text


def _source_text(cell: dict) -> str:
    """Return a notebook cell source as one string."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return ""


def _set_source(cell: dict, text: str) -> None:
    """Replace a notebook cell source while preserving line boundaries."""
    cell["source"] = [line for line in text.splitlines(keepends=True)]


def _is_url_corpus_dataset_load_cell(text: str) -> bool:
    """Identify a legacy dataset-loading cell requiring split-aware code."""
    return (
        "pd.read_csv" in text
        and "dataset_full" in text
        and "dataset_optimized" in text
        and "\ndatasets = {" in text
        and ("df_full" in text or "df_original" in text)
    )


def _is_url_corpus_data_split_cell(text: str) -> bool:
    """Identify a legacy feature-split cell."""
    return (
        "data_splits = {}" in text
        and "for name," in text
        and "datasets.items()" in text
        and ("df.iloc[:, :-1]" in text or "feature_columns" in text)
    )


def _is_url_corpus_train_test_cell(text: str) -> bool:
    """Identify a legacy random train/test cell."""
    return (
        "train_test_split(" in text
        and (
            "for name in data_splits.keys()" in text
            or "for name, data in data_splits.items()" in text
            or "train_test_data = {}" in text
        )
        and "X_train" in text
        and "X_test" in text
        and "y_train" in text
        and "y_test" in text
    )


def _apply_split_aware_cells(cells: list[dict]) -> None:
    """Replace legacy data cells with deterministic predefined split cells."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        text = _source_text(cell)
        if _is_url_corpus_dataset_load_cell(text):
            _set_source(cell, SPLIT_AWARE_LOAD_CELL)
        elif _is_url_corpus_data_split_cell(text):
            _set_source(cell, SPLIT_AWARE_DATA_SPLITS_CELL)
        elif _is_url_corpus_train_test_cell(text):
            _set_source(cell, SPLIT_AWARE_TRAIN_TEST_CELL)


def _apply_phase_testing_v4(cells, source_name):
    """Use staged quality-v4 payloads in the URL-corpus phase notebook."""
    if source_name != "6_comprehensive_model_testing.ipynb":
        return
    for cell in cells:
        text = _rewrite_phase_paths(_source_text(cell))
        if "model_configs = [" in text:
            text = PHASE_MODEL_LOAD_CELL
        elif "def prepare_test_data(df):" in text:
            text = PHASE_PREDICTION_HELPERS_CELL
        text = _rewrite_phase_prediction_calls(text)
        _set_source(cell, text)


def _rewrite_phase_paths(text):
    """Point phase testing at staged models and the selected test frame."""
    text = text.replace(
        "MODELS_DIR = backend_root / 'models'",
        "MODELS_DIR = backend_root / 'models' / 'staged_url_corpus_v4'",
    )
    return text.replace(
        "dataset_optimized_test.csv", "dataset_selected_test.csv"
    )


def _rewrite_phase_prediction_calls(text):
    """Align each phase frame to its model payload before prediction."""
    text = text.replace(
        "X_test, y_test = prepare_test_data(dist_df)",
        "X_test, y_test = prepare_test_data("
        "dist_df, model_data['feature_names'])",
    )
    return text.replace(
        "predict_with_model(model, scaler, X_test)",
        "predict_with_model(model_data, X_test)",
    )


def _remove_url_only_unused_imports(cells: list[dict], source_name: str) -> None:
    unused_names = URL_UNUSED_MODEL_SELECTION_IMPORTS.get(source_name, set())
    if not unused_names:
        return

    prefix = "from sklearn.model_selection import "
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        _set_source(cell, _remove_import_lines(_source_text(cell), prefix, unused_names))


def _remove_import_lines(text, prefix, unused_names):
    """Remove configured unused names from model-selection imports."""
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith(prefix):
            names = [name.strip() for name in line[len(prefix):].strip().split(",")]
            retained = [name for name in names if name not in unused_names]
            if retained:
                line = prefix + ", ".join(retained) + "\n"
            else:
                line = ""
        lines.append(line)
    return "".join(lines)


def _rewrite_cell_source(source):
    """Rewrite either string or list-form notebook source."""
    if isinstance(source, str):
        return _rewrite_text(source)
    if isinstance(source, list):
        return _rewrite_text("".join(source)).splitlines(keepends=True)
    return source


def _clear_outputs(cell: dict) -> None:
    """Clear execution outputs and count from a code cell."""
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _clear_runtime_metadata(cell: dict) -> None:
    """Remove volatile Jupyter execution metadata."""
    metadata = cell.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("execution", None)


def _preserve_canonical_sources(rendered: dict, existing_path: Path) -> None:
    """Keep deliberate corrections in an already generated notebook."""
    if not existing_path.exists():
        return
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    by_id = {cell.get("id"): cell for cell in existing.get("cells", [])}
    for cell in rendered.get("cells", []):
        previous = by_id.get(cell.get("id"))
        if cell.get("id") in CANONICAL_CORRECTED_CELL_IDS and previous:
            cell["source"] = previous.get("source", [])


def _apply_threshold_calibration_reporting(cells: list[dict]) -> None:
    """Persist and display validation-only threshold calibration details."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        text = _source_text(cell)
        text = _add_calibration_to_evaluation(text)
        text = _add_calibration_to_payload(text)
        text = _add_calibration_output(text)
        _set_source(cell, text)


def _apply_logistic_payload_save(cells: list[dict]) -> None:
    """Replace legacy separate model/scaler saving with one complete payload."""
    for cell in cells:
        text = _source_text(cell)
        if "logistic_regression_scaler_full_url_corpus.pkl" in text:
            _set_source(cell, LOGISTIC_PAYLOAD_SAVE_CELL)


def _add_calibration_to_evaluation(text: str) -> str:
    """Store calibration diagnostics beside the selected threshold."""
    marker = "        'decision_threshold': decision_threshold,\n"
    addition = marker + "        'threshold_calibration': threshold_info,\n"
    if "threshold_info = calibrate_threshold(" in text and addition not in text:
        return text.replace(marker, addition)
    return text


def _add_calibration_to_payload(text: str) -> str:
    """Include calibration provenance in serialized model payloads."""
    marker = "        'decision_threshold': evaluation_results[dataset_name]['decision_threshold'],\n"
    addition = (
        marker
        + "        'threshold_calibration': "
        + "evaluation_results[dataset_name]['threshold_calibration'],\n"
    )
    if addition in text:
        return text
    return text.replace(marker, addition)


def _add_calibration_output(text: str) -> str:
    """Print validation metrics that justify the frozen decision threshold."""
    marker = (
        '    print(f"  • Próg skalibrowany na walidacji: '
        "{decision_threshold:.4f}\")\n"
    )
    if marker not in text or "Walidacyjny FPR przy wybranym progu" in text:
        return text
    return text.replace(marker, _calibration_output_lines(marker))


def _calibration_output_lines(marker: str) -> str:
    """Return professional notebook messages describing calibration."""
    return (
        marker
        + '    print(f"  • Walidacyjny FPR przy wybranym progu: '
        "{threshold_info['validation_fpr']:.2%}\")\n"
        + '    print(f"  • Walidacyjna czułość przy wybranym progu: '
        "{threshold_info['validation_recall']:.2%}\")\n"
        + '    print(f"  • Docelowy FPR kalibracji: '
        "{threshold_info['target_fpr']:.2%}\")\n"
        + '    calibration_status = "spełniony" if '
        "threshold_info['safety_margin_satisfied'] else "
        '"niespełniony — zastosowano najniższy FPR zachowujący wymaganą czułość"\n'
        + '    print(f"  • Margines bezpieczeństwa kalibracji: {calibration_status}")\n'
    )


def render_notebook(src_path: Path) -> str:
    """Render one URL-corpus notebook from its original source."""
    nb = _load_notebook(src_path)
    cells = nb.get("cells", [])
    _prepare_cells(cells, src_path.name)
    _ensure_bootstrap(cells)
    return json.dumps(nb, ensure_ascii=False, indent=2) + "\n"


def _prepare_cells(cells, source_name):
    """Rewrite sources, clear runtime state and apply split replacements."""
    for cell in cells:
        cell["source"] = _rewrite_cell_source(cell.get("source", []))
        _clear_outputs(cell)
        _clear_runtime_metadata(cell)
    _apply_split_aware_cells(cells)
    _apply_phase_testing_v4(cells, source_name)
    _remove_url_only_unused_imports(cells, source_name)


def _ensure_bootstrap(cells):
    """Insert the portable backend bootstrap when it is missing."""
    has_bootstrap = any(
        isinstance(cell, dict)
        and cell.get("cell_type") == "code"
        and "backend_root =" in _source_text(cell)
        for cell in cells
    )
    if not has_bootstrap:
        cells.insert(0, BOOTSTRAP_CELL)


def _load_notebook(path):
    """Load notebook JSON from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_runtime_state(notebook_text: str) -> str:
    """Return deterministic notebook JSON while ignoring live Jupyter outputs."""
    notebook = json.loads(notebook_text)
    for cell in notebook.get("cells", []):
        _clear_outputs(cell)
        _clear_runtime_metadata(cell)
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"


def preserve_matching_runtime_state(
    rendered_text: str, existing_text: str
) -> str:
    """Keep cached outputs by stable cell id, without replacing newer outputs."""
    rendered = json.loads(rendered_text)
    existing = json.loads(existing_text)
    existing_by_id = {
        cell.get("id"): cell
        for cell in existing.get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("id")
    }

    for cell in rendered.get("cells", []):
        _restore_cell_runtime(cell, existing_by_id)

    return json.dumps(rendered, ensure_ascii=False, indent=2) + "\n"


def _restore_cell_runtime(cell, existing_by_id):
    """Restore cached output for one unchanged code cell."""
    if cell.get("cell_type") != "code" or not cell.get("id"):
        return
    if cell.get("outputs") or cell.get("execution_count") is not None:
        return
    previous = existing_by_id.get(cell["id"])
    if previous and previous.get("source") == cell.get("source"):
        cell["outputs"] = previous.get("outputs", [])
        cell["execution_count"] = previous.get("execution_count")


def convert_notebook(src_path: Path, dst_path: Path, *, check: bool = False, preserve_runtime: bool = True) -> bool:
    """Generate one notebook and optionally compare it to disk."""
    rendered = _render_target(src_path, dst_path)
    if check:
        if not dst_path.exists():
            return False
        existing = normalize_runtime_state(dst_path.read_text(encoding="utf-8"))
        return existing == rendered

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    output = rendered
    if preserve_runtime and dst_path.exists():
        output = preserve_matching_runtime_state(
            rendered, dst_path.read_text(encoding="utf-8")
        )
    dst_path.write_text(output, encoding="utf-8")
    return True


def _render_target(src_path, dst_path):
    """Render a notebook and preserve canonical source corrections."""
    notebook = json.loads(render_notebook(src_path))
    _preserve_canonical_sources(notebook, dst_path)
    _ensure_itertools_import(notebook)
    _ensure_used_plot_helpers(notebook.get("cells", []))
    _ensure_threshold_calibration_imports(notebook.get("cells", []))
    if src_path.name.startswith("1_logistic_regression"):
        _apply_logistic_payload_save(notebook.get("cells", []))
    _apply_threshold_calibration_reporting(notebook.get("cells", []))
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"


def _ensure_itertools_import(notebook):
    """Import itertools when a preserved grid-search cell requires it."""
    cells = notebook.get("cells", [])
    code = "\n".join(_source_text(cell) for cell in cells)
    if "itertools.product(" not in code or "import itertools\n" in code:
        return
    for cell in cells:
        text = _source_text(cell)
        if "import importlib\n" in text:
            _set_source(
                cell, text.replace(
                    "import importlib\n",
                    "import importlib\nimport itertools\n",
                    1,
                )
            )
            return


def _ensure_used_plot_helpers(cells):
    """Import shared plot helpers referenced by preserved notebook cells."""
    source = "\n".join(_source_text(cell) for cell in cells)
    if "remove_dataset_figure(" not in source:
        return
    for cell in cells:
        text = _source_text(cell)
        marker = "from utils.notebook_plot_style import (\n"
        if marker in text and "    remove_dataset_figure,\n" not in text:
            _set_source(
                cell, text.replace(
                    marker, marker + "    remove_dataset_figure,\n", 1
                )
            )
            return


def _ensure_threshold_calibration_imports(cells):
    """Import calibration helpers whenever generated cells reference them."""
    source = "\n".join(_source_text(cell) for cell in cells)
    marker = "from utils.threshold_calibration import ("
    if "calibrate_threshold(" not in source or marker in source:
        return
    for cell in cells:
        text = _source_text(cell)
        insertion = "from utils.feature_names import get_readable_name\n"
        if insertion in text:
            _set_source(
                cell, text.replace(
                    insertion, _calibration_import() + insertion, 1
                )
            )
            return


def _calibration_import() -> str:
    """Return imports required by validation-only threshold calibration."""
    return (
        "from utils.threshold_calibration import (\n"
        "    DEFAULT_MAX_FALSE_POSITIVE_RATE,\n"
        "    DEFAULT_MIN_RECALL,\n"
        "    calibrate_threshold,\n"
        ")\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse notebook generator options."""
    parser = argparse.ArgumentParser(
        description="Generate URL-corpus model notebooks from original notebooks."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when generated notebooks differ from files on disk.",
    )
    parser.add_argument(
        "--clean-runtime",
        action="store_true",
        help="Explicitly remove cached outputs and execution counts while generating.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate or check every URL-corpus notebook deterministically."""
    args = parse_args()
    backend_dir = Path(__file__).resolve().parent.parent
    notebooks_dir = backend_dir / "notebooks"
    source_dir = notebooks_dir / "original"
    target_dir = notebooks_dir / "url_corpus"

    generated, mismatched = _process_notebooks(args, source_dir, target_dir)
    _print_generated(generated)
    if mismatched:
        raise SystemExit(1)


def _process_notebooks(args, source_dir, target_dir):
    """Generate/check source notebooks and return successes and differences."""
    generated, mismatched = [], []
    for name in SOURCE_NOTEBOOKS:
        result = _process_one_notebook(args, source_dir, target_dir, name)
        if result is None:
            continue
        path, matches = result
        if args.check and not matches:
            mismatched.append(path)
            print(f"[DIFF] {path}")
        else:
            generated.append(path)
            print(f"[OK] {'Checked' if args.check else 'Generated'}: {path}")
    return generated, mismatched


def _process_one_notebook(args, source_dir, target_dir, name):
    """Convert one source notebook and report missing sources."""
    src = source_dir / name
    dst = target_dir / name.replace(".ipynb", "_url_corpus.ipynb")
    if not src.exists():
        print(f"[SKIP] Missing source notebook: {src}")
        return None
    matches = convert_notebook(src, dst, check=args.check, preserve_runtime=not args.clean_runtime)
    return dst, matches


def _print_generated(generated):
    """Print generated notebook paths."""
    print("\nGenerated notebooks:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
