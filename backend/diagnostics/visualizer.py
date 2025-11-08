import numpy as np
import matplotlib.pyplot as plt
import os


class OverfittingVisualizer:

    @staticmethod
    def plot_overfitting_analysis(model_name, learning_results, cv_results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analiza przeuczenia: {model_name.title()}', fontsize=16, fontweight='bold')

        OverfittingVisualizer._plot_learning_curves(axes[0, 0], learning_results)
        OverfittingVisualizer._plot_performance_gap(axes[0, 1], learning_results)
        OverfittingVisualizer._plot_cv_results(axes[1, 0], cv_results)
        OverfittingVisualizer._plot_cv_variance(axes[1, 1], cv_results)

        plt.tight_layout()

        os.makedirs('static/plots', exist_ok=True)
        plt.savefig(f'static/plots/{model_name}_overfitting_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📈 Zapisano wykres analizy przeuczenia: static/plots/{model_name}_overfitting_analysis.png")

        plt.show()

    @staticmethod
    def _plot_learning_curves(ax, learning_results):
        train_sizes = learning_results['train_sizes']
        train_mean = learning_results['train_mean']
        train_std = learning_results['train_std']
        val_mean = learning_results['val_mean']
        val_std = learning_results['val_std']

        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Wynik uczenia')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Wynik testowania')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        ax.set_xlabel('Rozmiar zbioru uczącego')
        ax.set_ylabel('Wynik ROC-AUC')
        ax.set_title('Krzywe uczenia')
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_performance_gap(ax, learning_results):
        train_sizes = learning_results['train_sizes']
        train_mean = learning_results['train_mean']
        val_mean = learning_results['val_mean']
        gap = train_mean - val_mean

        ax.plot(train_sizes, gap, 'o-', color='orange', linewidth=2)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Wysokie ryzyko (>0.05)')
        ax.axhline(y=0.02, color='yellow', linestyle='--', alpha=0.7, label='Umiarkowane ryzyko (>0.02)')
        ax.set_xlabel('Rozmiar zbioru uczącego')
        ax.set_ylabel('Różnica wydajności')
        ax.set_title('Różnica uczenie-testowanie')
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_cv_results(ax, cv_results):
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = {
            'accuracy': 'Dokładność',
            'precision': 'Precyzja',
            'recall': 'Czułość',
            'f1': 'F1',
            'roc_auc': 'ROC-AUC'
        }
        means = [cv_results[m]['mean'] for m in metrics]
        stds = [cv_results[m]['std'] for m in metrics]

        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Metryki')
        ax.set_ylabel('Wynik')
        ax.set_title('Wyniki walidacji krzyżowej')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

    @staticmethod
    def _plot_cv_variance(ax, cv_results):
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = {
            'accuracy': 'Dokładność',
            'precision': 'Precyzja',
            'recall': 'Czułość',
            'f1': 'F1',
            'roc_auc': 'ROC-AUC'
        }
        stds = [cv_results[m]['std'] for m in metrics]

        variance_colors = []
        for std in stds:
            if std > 0.05:
                variance_colors.append('red')
            elif std > 0.02:
                variance_colors.append('orange')
            else:
                variance_colors.append('green')

        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, stds, color=variance_colors, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Wysoka wariancja')
        ax.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Umiarkowana wariancja')
        ax.set_xlabel('Metryki')
        ax.set_ylabel('Odchylenie standardowe')
        ax.set_title('Wariancja walidacji krzyżowej')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
