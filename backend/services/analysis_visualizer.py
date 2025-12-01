import os
from pathlib import Path
from typing import Dict, Any, List
from io import BytesIO
import base64
import numpy as np


def _configure_matplotlib_cache():
    cache_dir = Path(os.environ.get('MPLCONFIGDIR', '/tmp/matplotlib'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache_dir)


_configure_matplotlib_cache()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class AnalysisVisualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid", palette="viridis")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10

    def _fig_to_base64(self, fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _extract_probabilities(self, predictions: List[Dict[str, Any]]) -> tuple:
        valid = [p for p in predictions if 'error' not in p]
        models = [p['model'] for p in valid]
        safe = [p['probabilities']['safe'] * 100 for p in valid]
        phishing = [p['probabilities']['phishing'] * 100 for p in valid]
        return models, safe, phishing

    def _add_bar_labels(self, ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8
            )

    def create_probability_chart(
        self,
        predictions: List[Dict[str, Any]]
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        models, safe_probs, phishing_probs = self._extract_probabilities(predictions)

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, safe_probs, width, label='Safe', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, phishing_probs, width, label='Phishing', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Probability (%)', fontweight='bold')
        ax.set_title('Phishing Detection Probability by Model', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')

        self._add_bar_labels(ax, bars1)
        self._add_bar_labels(ax, bars2)

        return self._fig_to_base64(fig)

    def create_voting_chart(self, prediction: Dict[str, Any]) -> str:
        if 'voting' not in prediction:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))

        voting = prediction['voting']
        sizes = [voting['safe'], voting['phishing']]
        labels = ['Safe', 'Phishing']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        ax.set_title(
            f'Random Forest Tree Voting\n'
            f'({voting["total_trees"]} trees)',
            fontweight='bold', fontsize=14, pad=15
        )

        return self._fig_to_base64(fig)

    def create_feature_importance_chart(
        self,
        prediction: Dict[str, Any]
    ) -> str:
        if 'top_features' not in prediction:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        features = prediction['top_features']
        names = [f['name'] for f in features]
        importances = [f['importance'] * 100 for f in features]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, importances, color='#3498db', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance (%)', fontweight='bold')
        ax.set_title('Top 10 Most Important Features', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for i, v in enumerate(importances):
            ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=8)

        return self._fig_to_base64(fig)

    def create_ensemble_voting_chart(
        self,
        ensemble_result: Dict[str, Any]
    ) -> str:
        fig, ax = plt.subplots(figsize=(8, 8))

        voting = ensemble_result['models_voting']
        sizes = [voting['safe'], voting['phishing']]
        labels = ['Safe', 'Phishing']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        ax.set_title(
            f'Ensemble Model Voting\n'
            f'({voting["total"]} models)',
            fontweight='bold', fontsize=14, pad=15
        )

        return self._fig_to_base64(fig)

    def _find_rf_prediction(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return next(
            (p for p in predictions if p.get('model') == 'random_forest'),
            None
        )

    def _add_rf_charts(self, report: Dict[str, Any], rf_prediction: Dict[str, Any]):
        voting_chart = self.create_voting_chart(rf_prediction)
        if voting_chart:
            report['charts']['voting'] = voting_chart

        importance_chart = self.create_feature_importance_chart(rf_prediction)
        if importance_chart:
            report['charts']['feature_importance'] = importance_chart

    def create_comprehensive_report(
        self,
        url: str,
        predictions: List[Dict[str, Any]],
        ensemble_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        report = {'url': url, 'charts': {}}
        report['charts']['probabilities'] = self.create_probability_chart(predictions)

        rf_prediction = self._find_rf_prediction(predictions)
        if rf_prediction:
            self._add_rf_charts(report, rf_prediction)

        if ensemble_result:
            ensemble_chart = self.create_ensemble_voting_chart(ensemble_result)
            if ensemble_chart:
                report['charts']['ensemble_voting'] = ensemble_chart

        return report
