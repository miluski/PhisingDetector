import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score,
    StratifiedKFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class OverfittingAnalyzer:

    def __init__(self, data_path='data/dataset_optimized.csv'):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        print(f"✓ Załadowano zbiór danych: {self.X.shape[0]:,} próbek, {self.X.shape[1]} cech")

    def load_model(self, model_name):
        model_path = f'models/{model_name}_model.pkl'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data['model'], model_data.get('scaler', None), model_data.get('metrics', {})

    def calculate_learning_curves(self, model, model_name, use_scaler=False):
        print(f"\n📊 Generowanie krzywych uczenia dla {model_name}...")

        X_data = self.X
        if use_scaler:
            X_data = self.scaler.fit_transform(self.X)
        else:
            X_data = self.X.values if hasattr(self.X, 'values') else self.X

        y_data = self.y.values if hasattr(self.y, 'values') else self.y

        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_data, y_data,
            train_sizes=train_sizes,
            cv=5,
            scoring='roc_auc',
            n_jobs=1,
            random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        final_gap = train_mean[-1] - val_mean[-1]
        max_gap = np.max(train_mean - val_mean)

        print(f"  Końcowy wynik uczenia: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
        print(f"  Końcowy wynik testowania: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
        print(f"  Luka wydajności: {final_gap:.4f}")
        print(f"  Maksymalna luka: {max_gap:.4f}")

        if final_gap > 0.05:
            print(f"  ⚠️  WYSOKIE ryzyko przeuczenia (luka > 0.05)")
        elif final_gap > 0.02:
            print(f"  ⚠️  UMIARKOWANE ryzyko przeuczenia (luka > 0.02)")
        else:
            print(f"  ✓ NISKIE ryzyko przeuczenia (luka ≤ 0.02)")

        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std,
            'final_gap': final_gap,
            'max_gap': max_gap
        }

    def cross_validation_analysis(self, model, model_name, use_scaler=False):
        print(f"\n🔄 Analiza walidacji krzyżowej dla {model_name}...")

        X_data = self.X
        if use_scaler:
            X_data = self.scaler.fit_transform(self.X)
        else:
            X_data = self.X.values if hasattr(self.X, 'values') else self.X

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        y_data = self.y.values if hasattr(self.y, 'values') else self.y

        for train_idx, val_idx in skf.split(X_data, y_data):
            X_train_fold, X_val_fold = X_data[train_idx], X_data[val_idx]
            y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_fold, y_train_fold)

            y_pred = model_clone.predict(X_val_fold)
            y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]

            metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            metrics['precision'].append(precision_score(y_val_fold, y_pred))
            metrics['recall'].append(recall_score(y_val_fold, y_pred))
            metrics['f1'].append(f1_score(y_val_fold, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))

        results = {}
        for metric, scores in metrics.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results[metric] = {
                'mean': mean_score,
                'std': std_score,
                'scores': scores
            }

            metric_names = {
                'accuracy': 'DOKŁADNOŚĆ',
                'precision': 'PRECYZJA',
                'recall': 'CZUŁOŚĆ',
                'f1': 'F1',
                'roc_auc': 'ROC-AUC'
            }
            print(f"  {metric_names.get(metric, metric.upper())}: {mean_score:.4f} (±{std_score:.4f})")

            if std_score > 0.05:
                print(f"    ⚠️  WYSOKA wariancja w {metric}")
            elif std_score > 0.02:
                print(f"    ⚠️  UMIARKOWANA wariancja w {metric}")
            else:
                print(f"    ✓ NISKA wariancja w {metric}")

        return results

    def train_test_gap_analysis(self, model, model_name, use_scaler=False):
        print(f"\n Analiza różnicy między uczeniem a testowaniem dla {model_name}...")

        X_train_data = self.X_train
        X_test_data = self.X_test
        if use_scaler:
            X_train_data = self.scaler.fit_transform(self.X_train)
            X_test_data = self.scaler.transform(self.X_test)
        else:
            X_train_data = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            X_test_data = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test

        y_train_data = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
        y_test_data = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test

        y_train_pred = model.predict(X_train_data)
        y_test_pred = model.predict(X_test_data)

        train_acc = accuracy_score(y_train_data, y_train_pred)
        test_acc = accuracy_score(y_test_data, y_test_pred)
        gap = train_acc - test_acc

        print(f"  Dokładność uczenia: {train_acc:.4f}")
        print(f"  Dokładność testowania:    {test_acc:.4f}")
        print(f"  Różnica:               {gap:.4f}")

        if gap > 0.05:
            print(f"    ⚠️  DUŻA różnica - możliwe przeuczenie!")
        elif gap > 0.02:
            print(f"    ⚠️  UMIARKOWANA różnica")
        else:
            print(f"    ✓ MAŁA różnica - dobra generalizacja")

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap
        }

    def calculate_overfitting_score(self, learning_results, cv_results, gap_results):
        overfitting_score = 0

        if learning_results['final_gap'] > 0.05:
            overfitting_score += 3
        elif learning_results['final_gap'] > 0.02:
            overfitting_score += 2

        avg_cv_std = np.mean([cv_results[m]['std'] for m in cv_results])
        if avg_cv_std > 0.05:
            overfitting_score += 3
        elif avg_cv_std > 0.02:
            overfitting_score += 2

        if gap_results['avg_gap'] > 0.05:
            overfitting_score += 3
        elif gap_results['avg_gap'] > 0.02:
            overfitting_score += 2

        return overfitting_score
