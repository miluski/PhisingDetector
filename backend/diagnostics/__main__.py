#!/usr/bin/env python3
"""
Diagnostics Package CLI
Unified interface for overfitting diagnosis and model analysis
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.reporter import OverfittingReporter


def quick_diagnosis():
    """Quick diagnosis of overfitting across all models"""
    reporter = OverfittingReporter()
    models = ['logistic_regression', 'random_forest', 'xgboost', 'svm', 'mlp']

    print("🚀 Szybka diagnoza przeuczenia")
    print("="*50)

    results = {}

    for model_name in models:
        try:
            print(f"\n📊 Analizowanie {model_name}...")
            result = reporter.generate_overfitting_report(model_name, detailed=False)
            if result:
                results[model_name] = result['overfitting_score']
        except Exception as e:
            print(f"⚠️ Nie można przeanalizować {model_name}: {e}")

    print(f"\n{'='*50}")
    print("📋 PODSUMOWANIE PRZEUCZENIA")
    print(f"{'='*50}")

    if results:
        for model, score in sorted(results.items(), key=lambda x: x[1]):
            status = "✅ Dobry" if score < 2 else "⚠️ Umiarkowany" if score < 4 else "❌ Wysoki"
            print(f"{model:<20} Wynik: {score}/9  {status}")

        best_model = min(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Najlepszy model (najniższe przeuczenie): {best_model[0]} (wynik: {best_model[1]})")
    else:
        print("Nie znaleziono modeli. Najpierw wyucz modele, uruchamiając notebooki.")


def detailed_diagnosis(model_name=None, all_models=False, detailed=True):
    """Detailed diagnosis with plots and analysis"""
    reporter = OverfittingReporter()
    available_models = ['logistic_regression', 'random_forest', 'xgboost', 'svm', 'mlp']

    if all_models:
        print("🔍 Analizowanie wszystkich dostępnych modeli...")
        for model in available_models:
            try:
                reporter.generate_overfitting_report(model, detailed)
                print("\n" + "="*80 + "\n")
            except FileNotFoundError:
                print(f"⚠️  Model {model} nie znaleziony, pomijam...")

    elif model_name:
        if model_name in available_models:
            reporter.generate_overfitting_report(model_name, detailed)
        else:
            print(f"❌ Nieznany model: {model_name}")
            print(f"Dostępne modele: {', '.join(available_models)}")
    else:
        print("❌ Proszę określić --model <nazwa_modelu> lub --all")
        print(f"Dostępne modele: {', '.join(available_models)}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Diagnozuj przeuczenie w modelach detekcji phishingu',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  %(prog)s --quick                    # Szybka diagnoza wszystkich modeli
  %(prog)s --model mlp                # Szczegółowa analiza modelu MLP
  %(prog)s --all --detailed           # Szczegółowa analiza wszystkich modeli
  %(prog)s --model svm --detailed     # Szczegółowa analiza SVM z wykresami
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Szybka diagnoza wszystkich modeli (bez wykresów)')
    parser.add_argument('--model', type=str,
                       help='Model do analizy (logistic_regression, random_forest, xgboost, svm, mlp)')
    parser.add_argument('--all', action='store_true',
                       help='Analizuj wszystkie dostępne modele')
    parser.add_argument('--detailed', action='store_true',
                       help='Generuj szczegółowe wykresy i analizę')

    args = parser.parse_args()

    if args.quick:
        quick_diagnosis()
    elif args.model or args.all:
        detailed_diagnosis(args.model, args.all, args.detailed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
