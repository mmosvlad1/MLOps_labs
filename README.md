# Лабораторна робота №4. CI/CD для ML-проєктів: автоматизація тестування та звітності з GitHub Actions та CML

## 1. Мета роботи

1. Зрозуміти принципи Continuous Integration (CI) та Continuous Delivery (CD) в контексті Machine Learning (код, дані, конфігурації, артефакти моделі).
2. Навчитися створювати автоматизовані workflows за допомогою GitHub Actions.
3. Опанувати інструмент CML (Continuous Machine Learning) для автоматичної генерації звітів по експериментах у Pull Request.
4. Впровадити автоматичне тестування коду, даних та артефактів (pytest) у процес розробки.
5. Реалізувати концепцію Quality Gate: формалізоване правило (порог/критерій), яке автоматично визначає, чи приймається модель/зміна.

## 2. Виконані завдання

1. ✅ Репозиторій на GitHub з ML-проєктом (продовження ЛР2–ЛР3).
2. ✅ Додано набір тестів (Unit/Smoke Tests):
   - **Pre-train:** валідація якості/структури даних (raw і prepared), наявність колонок, діапазони значень, відсутність null, обидва класи.
   - **Post-train:** перевірка артефактів (model.pkl, metrics.json, confusion_matrix.png), цілісність моделі (predict/predict_proba), Quality Gate за метриками (AUPRC, F1, recall ≥ порогу).
3. ✅ Створено `.github/workflows/cml.yaml`.
4. ✅ Workflow налаштовано для `push` та `pull_request` на гілку `main`:
   - встановлення залежностей (pip),
   - лінтинг (flake8, black),
   - завантаження датасету (Kaggle),
   - підготовка даних з підвибіркою (CI_MAX_ROWS=50000),
   - pre-train тести,
   - тренування (python -m src.train),
   - post-train тести (Quality Gate),
   - формування та публікація CML-звіту в PR.

## 3. Структура проєкту

```
.github/workflows/
└── cml.yaml              # CI workflow (lint, train, test, CML report)
tests/
├── test_pre_train.py     # Валідація даних до тренування
└── test_post_train.py    # Артефакти + Quality Gate після тренування
src/
├── prepare.py            # Підготовка даних (підтримує --max-rows для CI)
├── train.py              # Тренування + збереження артефактів у data/models/
└── optimize.py
data/models/              # Артефакти після тренування
├── model.pkl
├── metrics.json
└── confusion_matrix.png
```

## 4. Pre-train тести

**Raw data (TestRawData):** наявність файлу, обов’язкові колонки (V1–V28, Time, Amount, Class), бінарний target (0/1), наявність fraud-кейсів, non-negative Amount.

**Prepared data (TestPreparedData):** наявність train/test, колонки (V1–V28, hour_of_day, Amount_scaled, Class), відсутність null, train > test за розміром, hour_of_day ∈ [0, 23], обидва класи в train, однакова схема train і test.

## 5. Post-train тести

**Артефакти (TestArtifactsExist):** model.pkl, metrics.json, confusion_matrix.png — існують і не порожні.

**Модель (TestModelIntegrity):** модель завантажується, має `predict` та `predict_proba`.

**Quality Gate (TestMetricsQualityGate):** metrics.json містить test_auprc, test_f1, test_recall, test_precision; пороги: AUPRC ≥ 0.10, F1 ≥ 0.10, recall ≥ 0.10; значення в діапазоні [0, 1].

## 6. GitHub Actions workflow

**Тригери:** `push` та `pull_request` на `main`.

**Кроки:**

| Крок | Опис |
|------|------|
| checkout | Клон репозиторію |
| Set up Python | Python 3.11, pip cache |
| Install dependencies | `pip install -r requirements.txt` |
| Lint (flake8) | Критичні помилки (E9, F63, F7, F82) у src/, tests/ |
| Format check (black) | Перевірка форматування |
| Download dataset (Kaggle) | Завантаження creditcard.csv через kagglehub (секрети KAGGLE_USERNAME, KAGGLE_KEY) |
| Prepare data | `--max-rows 50000` для прискорення CI |
| Pre-train tests | `pytest tests/test_pre_train.py -v` |
| Train model | XGBoost, n_estimators=100, max_depth=10, --model-dir data/models |
| Post-train tests | `pytest tests/test_post_train.py -v` |
| Setup CML | iterative/setup-cml@v2 (лише для PR) |
| Create CML report | Метрики (JSON) + confusion matrix, коментар у PR |

## 7. CML-звіт у Pull Request

При `pull_request` workflow формує markdown-звіт:
- **Model Metrics** — вміст metrics.json у блоці коду
- **Confusion Matrix** — вбудоване зображення
- Підпис: _Trained on CI subset: 50000 rows_

Звіт публікується як коментар до PR командою `cml comment create report.md`.

## 8. Артефакти моделі

Скрипт `src/train.py` при передачі `--model-dir` зберігає:
- `model.pkl` — серіалізована модель (joblib)
- `metrics.json` — train/test метрики (recall, precision, F1, AUPRC)
- `confusion_matrix.png` — візуалізація матриці помилок

Це дозволяє перевіряти артефакти в CI та формувати CML-звіт.

## 9. Відтворюваність

- **Seed:** 42 у скриптах (random_state, stratify).
- **Дані в CI:** датасет Kaggle (mlg-ulb/creditcardfraud), підвибірка 50000 рядків зі стратифікацією за Class.
- **Версія коду:** Git commit hash через checkout.
- **Конфігурація:** фіксовані параметри тренування в workflow (xgboost, n_estimators=100, max_depth=10, learning_rate=0.1).

## 10. Висновки

У межах лабораторної роботи інтегровано CI/CD для ML-проєкту з використанням GitHub Actions та CML. Реалізовано дві групи тестів: pre-train (валідація структури та якості даних) та post-train (наявність артефактів, цілісність моделі, Quality Gate за метриками AUPRC, F1, recall). Workflow виконує лінтинг, завантаження даних з Kaggle, підготовку з підвибіркою для прискорення CI, тренування XGBoost та публікацію звіту в PR. Quality Gate забезпечує, що модель не пройде CI при падінні метрик нижче заданих порогів.
