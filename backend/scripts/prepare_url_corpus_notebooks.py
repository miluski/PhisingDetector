from __future__ import annotations

import json
from pathlib import Path


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


BOOTSTRAP_CELL = {
    "cell_type": "code",
    "metadata": {
        "language": "python"
    },
    "source": [
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "cwd = Path.cwd().resolve()\n",
        "if cwd.name == 'url_corpus' and cwd.parent.name == 'notebooks':\n",
        "    os.chdir(cwd.parent)\n",
        "\n",
        "nb_dir = Path.cwd().resolve()\n",
        "backend_root = nb_dir.parent if nb_dir.name == 'notebooks' else nb_dir\n",
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
        "# Always generate missing-values artifacts, even when -1 does not occur in top features.\n",
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
        "        hovertemplate='<b>%{x}</b><br>Nie występuje w URL: %{y:.1f}%<br>Liczba: %{customdata}<extra></extra>',\n",
        "        customdata=missing_df['Bez phishingu (count)']\n",
        "    ))\n",
        "    fig_missing.add_trace(go.Bar(\n",
        "        name='Phishing',\n",
        "        x=missing_df['Cecha'],\n",
        "        y=missing_df['Phishing'],\n",
        "        marker_color='#EF553B',\n",
        "        hovertemplate='<b>%{x}</b><br>Nie występuje w URL: %{y:.1f}%<br>Liczba: %{customdata}<extra></extra>',\n",
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


def _rewrite_text(text: str) -> str:
    updated = text
    updated = updated.replace(
        "sys.path.append('..')",
        "cwd = Path.cwd().resolve(); os.chdir(cwd.parent) if cwd.name == 'url_corpus' and cwd.parent.name == 'notebooks' else None; nb_dir = Path.cwd().resolve(); backend_root = nb_dir.parent if nb_dir.name == 'notebooks' else nb_dir; sys.path.insert(0, str(backend_root)) if str(backend_root) not in sys.path else None"
    )
    updated = updated.replace("../data/dataset_full.csv", "../data/url_corpus/dataset_full.csv")
    updated = updated.replace("../data/dataset_optimized.csv", "../data/url_corpus/dataset_optimized.csv")
    updated = updated.replace("../static/statistical_analysis", "../static/statistical_analysis_url_corpus")
    updated = updated.replace("../static/plots/", "../static/plots_url_corpus/")
    updated = updated.replace("../data/phase_testing_results", "../data/url_corpus/phase_testing_results")
    updated = updated.replace("../data/logistic_regression_comparison.csv", "../data/url_corpus/logistic_regression_comparison.csv")
    updated = updated.replace("../data/random_forest_comparison.csv", "../data/url_corpus/random_forest_comparison.csv")
    updated = updated.replace("../data/xgboost_comparison.csv", "../data/url_corpus/xgboost_comparison.csv")
    updated = updated.replace("../data/svm_comparison.csv", "../data/url_corpus/svm_comparison.csv")
    updated = updated.replace("../data/mlp_comparison.csv", "../data/url_corpus/mlp_comparison.csv")
    updated = updated.replace(
        "df = pd.read_csv('../data/url_corpus/dataset_full.csv')",
        "dataset_path = '../data/url_corpus/dataset_full.csv'; dataset_path = dataset_path if os.path.exists(dataset_path) else '../data/url_corpus/dataset_optimized.csv'; print(f'Using dataset: {dataset_path}'); df = pd.read_csv(dataset_path)"
    )
    updated = updated.replace("../data/test", "../data/url_corpus/test")
    updated = updated.replace("_{}.pkl", "_{}_url_corpus.pkl")
    updated = updated.replace("_{dataset_type}.pkl", "_{dataset_type}_url_corpus.pkl")
    updated = updated.replace("_{file_suffix}.pkl", "_{file_suffix}_url_corpus.pkl")
    updated = updated.replace("_full.pkl", "_full_url_corpus.pkl")
    updated = updated.replace("_optimized.pkl", "_optimized_url_corpus.pkl")
    return updated


def _rewrite_cell_source(source):
    if isinstance(source, str):
        return _rewrite_text(source)
    if isinstance(source, list):
        return [_rewrite_text(line) for line in source]
    return source


def _clear_outputs(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def convert_notebook(src_path: Path, dst_path: Path) -> None:
    with src_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    for cell in cells:
        cell["source"] = _rewrite_cell_source(cell.get("source", []))
        _clear_outputs(cell)

    has_bootstrap = any(
        isinstance(cell, dict)
        and cell.get("cell_type") == "code"
        and any("Backend root in sys.path" in str(line) for line in cell.get("source", []))
        for cell in cells
    )
    if not has_bootstrap:
        cells.insert(0, BOOTSTRAP_CELL)

    if src_path.name == "0_comprehensive_statistical_analysis.ipynb":
        has_missing_helper = any(
            isinstance(cell, dict)
            and cell.get("cell_type") == "code"
            and any("generate_missing_minus_one_artifacts" in str(line) for line in cell.get("source", []))
            for cell in cells
        )
        if not has_missing_helper:
            insert_after = -1
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code" and any(
                    "distribution_analysis = analyze_distributions(df_clean)" in str(line)
                    for line in cell.get("source", [])
                ):
                    insert_after = idx
                    break
            if insert_after >= 0:
                cells.insert(insert_after + 1, MISSING_MINUS_ONE_ALWAYS_CELL)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)


def main() -> None:
    backend_dir = Path(__file__).resolve().parent.parent
    notebooks_dir = backend_dir / "notebooks"
    target_dir = notebooks_dir / "url_corpus"

    generated = []
    for name in SOURCE_NOTEBOOKS:
        src = notebooks_dir / name
        dst = target_dir / name.replace(".ipynb", "_url_corpus.ipynb")
        if not src.exists():
            print(f"[SKIP] Missing source notebook: {src}")
            continue
        convert_notebook(src, dst)
        generated.append(dst)
        print(f"[OK] Generated: {dst}")

    print("\nGenerated notebooks:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
