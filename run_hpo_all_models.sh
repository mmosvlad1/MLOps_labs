# HPO для всіх моделей: xgboost, random_forest, lightgbm по 50 trials кожна

set -e
python -m src.optimize hpo.run_all_models=true
