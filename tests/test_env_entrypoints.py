from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_wait_and_train_mlp_uses_dante_env_by_default() -> None:
    text = _read("scripts/wait_and_train_mlp.sh")
    assert 'ENV_NAME="dante_env"' in text
    assert 'conda run -n "${ENV_NAME}"' in text
    assert "python scripts/train_mlp.py" in text


def test_wait_and_run_baselines_uses_dante_env_by_default() -> None:
    text = _read("scripts/wait_and_run_baselines.sh")
    assert 'ENV_NAME="dante_env"' in text
    assert 'conda run -n "${ENV_NAME}"' in text


def test_check_ml_env_script_exists_and_checks_tensorflow_gpu() -> None:
    script = REPO_ROOT / "scripts" / "check_ml_env.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "tensorflow" in text
    assert "list_physical_devices(\"GPU\")" in text

def test_inverse_5deg_pipeline_defaults_use_quasi_exp_cpu_configs() -> None:
    text = _read("scripts/pipelines/run_inverse_5deg_autogate.sh")
    assert 'ENV_NAME="${ENV_NAME:-quasi_exp}"' in text
    assert 'CFG_2K="${CFG_2K:-configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu.yaml}"' in text
    assert 'CFG_100K="${CFG_100K:-configs/robot_rods_only_5deg_inverse_100k_paper_tf_cpu.yaml}"' in text


def test_inverse_5deg_tune_defaults_use_quasi_exp_cpu_configs() -> None:
    gate_text = _read("scripts/pipelines/tune_inverse_5deg_2k_gate.py")
    adaptive_text = _read("scripts/pipelines/tune_inverse_5deg_2k_adaptive.py")
    assert 'default="configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu.yaml"' in gate_text
    assert 'ap.add_argument("--env-name", default="quasi_exp")' in gate_text
    assert 'default="configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu_mt.yaml"' in adaptive_text
    assert 'ap.add_argument("--env-name", default="quasi_exp")' in adaptive_text

