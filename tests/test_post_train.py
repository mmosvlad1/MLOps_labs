"""Post-train checks: run after DVC pipeline to validate artifacts and model quality."""

import json
from pathlib import Path

import joblib
import pytest

MODEL_DIR = Path("data/models")
MODEL_PKL = MODEL_DIR / "model.pkl"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PNG = MODEL_DIR / "confusion_matrix.png"

# Quality gates
MIN_TEST_AUPRC = 0.10
MIN_TEST_F1 = 0.10
MIN_TEST_RECALL = 0.10


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------


class TestArtifactsExist:
    def test_model_pkl_exists(self):
        assert MODEL_PKL.exists(), f"Model artifact not found: {MODEL_PKL}"

    def test_metrics_json_exists(self):
        assert METRICS_JSON.exists(), f"Metrics file not found: {METRICS_JSON}"

    def test_confusion_matrix_exists(self):
        assert (
            CONFUSION_MATRIX_PNG.exists()
        ), f"Confusion matrix not found: {CONFUSION_MATRIX_PNG}"

    def test_model_pkl_not_empty(self):
        assert MODEL_PKL.stat().st_size > 0, f"{MODEL_PKL} is empty"

    def test_metrics_json_not_empty(self):
        assert METRICS_JSON.stat().st_size > 0, f"{METRICS_JSON} is empty"

    def test_confusion_matrix_not_empty(self):
        assert (
            CONFUSION_MATRIX_PNG.stat().st_size > 0
        ), f"{CONFUSION_MATRIX_PNG} is empty"


# ---------------------------------------------------------------------------
# Model artifact integrity
# ---------------------------------------------------------------------------


class TestModelIntegrity:
    @pytest.fixture(scope="class")
    def model(self):
        if not MODEL_PKL.exists():
            pytest.skip("model.pkl not found")
        return joblib.load(MODEL_PKL)

    def test_model_loadable(self, model):
        assert model is not None

    def test_model_has_predict(self, model):
        assert hasattr(model, "predict"), "Model missing predict() method"

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba"), "Model missing predict_proba() method"


# ---------------------------------------------------------------------------
# Metrics quality gates
# ---------------------------------------------------------------------------


class TestMetricsQualityGate:
    @pytest.fixture(scope="class")
    def metrics(self):
        if not METRICS_JSON.exists():
            pytest.skip("metrics.json not found")
        with open(METRICS_JSON) as f:
            return json.load(f)

    def test_metrics_json_valid(self, metrics):
        assert isinstance(metrics, dict), "metrics.json must be a JSON object"

    def test_required_metric_keys(self, metrics):
        required = {"test_auprc", "test_f1", "test_recall", "test_precision"}
        missing = required - set(metrics.keys())
        assert not missing, f"Missing keys in metrics.json: {missing}"

    def test_auprc_quality_gate(self, metrics):
        auprc = metrics["test_auprc"]
        assert (
            auprc >= MIN_TEST_AUPRC
        ), f"test_auprc={auprc:.4f} is below quality gate ({MIN_TEST_AUPRC})"

    def test_f1_quality_gate(self, metrics):
        f1 = metrics["test_f1"]
        assert (
            f1 >= MIN_TEST_F1
        ), f"test_f1={f1:.4f} is below quality gate ({MIN_TEST_F1})"

    def test_recall_quality_gate(self, metrics):
        recall = metrics["test_recall"]
        assert (
            recall >= MIN_TEST_RECALL
        ), f"test_recall={recall:.4f} is below quality gate ({MIN_TEST_RECALL})"

    def test_metrics_are_valid_probabilities(self, metrics):
        for key in ("test_auprc", "test_f1", "test_recall", "test_precision"):
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"
