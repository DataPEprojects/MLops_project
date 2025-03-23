import sys
import os
# Ajoute le répertoire parent du dossier tests au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.train_model import (
    run_logistic_regression_experiment,
    run_random_forest_experiment,
    run_catboost_experiment
)

def test_logistic_regression_experiment():
    data_path = os.path.join("data", "Loan_Data.csv")
    # Utilisation d'un nombre réduit de folds et une itération pour accélérer le test
    metrics = run_logistic_regression_experiment(data_path, feature_set="base", weighted=True, cv_folds=3, random_state=1)

    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {metric} out of bounds: {value}"

def test_random_forest_experiment():
    data_path = os.path.join("data", "Loan_Data.csv")
    # On utilise un nombre réduit d'arbres pour accélérer le test
    metrics = run_random_forest_experiment(data_path, feature_set="base", cv_folds=3, random_state=1, n_estimators=10)
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {metric} out of bounds: {value}"

def test_catboost_experiment():
    data_path = os.path.join("data", "Loan_Data.csv")
    # Réduire le nombre d'itérations pour accélérer le test
    metrics = run_catboost_experiment(data_path, feature_set="base", cv_folds=3, random_state=1, iterations=10)
    for metric, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {metric} out of bounds: {value}"

if __name__ == "__main__":
    test_logistic_regression_experiment()
    test_random_forest_experiment()
    test_catboost_experiment()
    print("All tests passed!")
