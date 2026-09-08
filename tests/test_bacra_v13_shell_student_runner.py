from pathlib import Path


def test_student_runner_locks_models_before_opening_sealed_shell() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/analysis/run_bacra_v13_shell_student.py").read_text()
    assert text.index('atomic_write_json(lock_stage / "model_lock.json"') < text.index(
        'sealed = pd.read_parquet('
    )
    assert "StudentStrategy.CHART_CONDITIONED" in text
    assert "automatic_chart_classifier_disabled" in text
    assert 'return 0 if bool(report.get("gate_pass")) else 2' in text
