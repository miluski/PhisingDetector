import numpy as np
from .analyzer import OverfittingAnalyzer
from .visualizer import OverfittingVisualizer


class OverfittingReporter:

    def __init__(self, data_path='data/dataset_optimized.csv'):
        self.analyzer = OverfittingAnalyzer(data_path)
        self.visualizer = OverfittingVisualizer()

    def generate_overfitting_report(self, model_name, detailed=False):
        print(f"\n{'='*60}")
        print(f"🔍 RAPORT DIAGNOZY PRZEUCZENIA: {model_name.upper()}")
        print(f"{'='*60}")

        try:
            model, scaler, stored_metrics = self.analyzer.load_model(model_name)
            use_scaler = scaler is not None

            print(f"✓ Załadowano model: {model.__class__.__name__}")
            print(f"✓ Używa skalowania: {use_scaler}")

            learning_results = self.analyzer.calculate_learning_curves(model, model_name, use_scaler)
            cv_results = self.analyzer.cross_validation_analysis(model, model_name, use_scaler)
            gap_results = self.analyzer.train_test_gap_analysis(model, model_name, use_scaler)

            if detailed:
                self.visualizer.plot_overfitting_analysis(model_name, learning_results, cv_results)

            self._print_overall_assessment(learning_results, cv_results, gap_results)

            return {
                'overfitting_score': self.analyzer.calculate_overfitting_score(
                    learning_results, cv_results, gap_results
                ),
                'learning_results': learning_results,
                'cv_results': cv_results,
                'gap_results': gap_results
            }

        except Exception as e:
            print(f"❌ Błąd podczas analizy {model_name}: {str(e)}")
            return None

    def _print_overall_assessment(self, learning_results, cv_results, gap_results):
        print(f"\n{'='*60}")
        print("🎯 OGÓLNA OCENA PRZEUCZENIA")
        print(f"{'='*60}")

        overfitting_score = 0

        if learning_results['final_gap'] > 0.05:
            overfitting_score += 3
            print("❌ Krzywa uczenia pokazuje WYSOKIE przeuczenie")
        elif learning_results['final_gap'] > 0.02:
            overfitting_score += 2
            print("⚠️  Krzywa uczenia pokazuje UMIARKOWANE przeuczenie")
        else:
            print("✅ Krzywa uczenia wygląda dobrze")

        avg_cv_std = np.mean([cv_results[m]['std'] for m in cv_results])
        if avg_cv_std > 0.05:
            overfitting_score += 3
            print("❌ WYSOKA wariancja w walidacji krzyżowej")
        elif avg_cv_std > 0.02:
            overfitting_score += 2
            print("⚠️  UMIARKOWANA wariancja w walidacji krzyżowej")
        else:
            print("✅ Niska wariancja w walidacji krzyżowej")

        if gap_results['avg_gap'] > 0.05:
            overfitting_score += 3
            print("❌ WYSOKA różnica wydajności uczenie-testowanie")
        elif gap_results['avg_gap'] > 0.02:
            overfitting_score += 2
            print("⚠️  UMIARKOWANA różnica wydajności uczenie-testowanie")
        else:
            print("✅ Niska różnica wydajności uczenie-testowanie")

        print(f"\nOSTATECZNY WERDYKT:")
        if overfitting_score >= 7:
            print("❌ POWAŻNE PRZEUCZENIE - Model wymaga regularyzacji")
        elif overfitting_score >= 4:
            print("⚠️  UMIARKOWANE PRZEUCZENIE - Rozważ poprawki")
        elif overfitting_score >= 2:
            print("⚠️  NIEWIELKIE PRZEUCZENIE - Monitoruj uważnie")
        else:
            print("✅ BRAK ZNACZĄCEGO PRZEUCZENIA - Model wygląda dobrze")

        print(f"\nWynik przeuczenia: {overfitting_score}/9")

        self._print_recommendations(overfitting_score)

    @staticmethod
    def _print_recommendations(overfitting_score):
        print(f"\n💡 REKOMENDACJE:")
        if overfitting_score >= 4:
            print("• Zwiększ parametry regularyzacji")
            print("• Zmniejsz złożoność modelu")
            print("• Zbierz więcej danych uczących")
            print("• Zastosuj selekcję cech")
            print("• Użyj metod ensemble")
        elif overfitting_score >= 2:
            print("• Monitoruj wydajność na nowych danych")
            print("• Rozważ niewielkie zwiększenie regularyzacji")
            print("• Waliduj na dodatkowych zbiorach testowych")
        else:
            print("• Obecna konfiguracja modelu wygląda dobrze")
            print("• Kontynuuj monitorowanie z nowymi danymi")
