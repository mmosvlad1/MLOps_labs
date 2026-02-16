# Порівняння TPE vs Random sampler

set -e
echo "1. HPO з TPE sampler (XGBoost, 20 trials)..."
python -m src.optimize hpo=tpe

echo ""
echo "2. HPO з Random sampler (XGBoost, 20 trials)..."
python -m src.optimize hpo=random
