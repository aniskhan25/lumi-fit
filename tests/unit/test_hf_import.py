import json
from pathlib import Path

from lumi_fit.adapters.hf_import import import_huggingface_snapshot
from lumi_fit.domain.models import ModelCapability, WeightDtype


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/hf"


def test_import_huggingface_snapshot_uses_immutable_revision_and_exact_size() -> None:
    model, artifact = import_huggingface_snapshot(
        config_path=FIXTURES / "qwen-config.json",
        api_path=FIXTURES / "qwen-api.json",
        safetensors_index_path=FIXTURES / "qwen-index.json",
    )

    assert model.model_id == "Qwen/Qwen3-8B"
    assert model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert model.architecture.total_parameters == 8_190_735_360
    assert model.architecture.kv_heads == 8
    assert ModelCapability.CHAT in model.capabilities
    assert artifact.total_bytes == 16_381_470_720
    assert artifact.weight_dtype is WeightDtype.BF16
    assert artifact.compatible_runtime_ids == ("vllm-laif",)


def test_import_derives_head_dimension_and_reads_modern_dtype(
    tmp_path: Path,
) -> None:
    config = json.loads((FIXTURES / "qwen-config.json").read_text(encoding="utf-8"))
    config.pop("head_dim")
    config["dtype"] = config.pop("torch_dtype")
    api = json.loads((FIXTURES / "qwen-api.json").read_text(encoding="utf-8"))
    api["tags"].extend(["code", "reasoning", "function-calling", "fi"])
    api["cardData"] = {"language": ["en", "zh"]}
    config_path = tmp_path / "config.json"
    api_path = tmp_path / "api.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    api_path.write_text(json.dumps(api), encoding="utf-8")

    model, _ = import_huggingface_snapshot(
        config_path=config_path,
        api_path=api_path,
        safetensors_index_path=FIXTURES / "qwen-index.json",
    )

    assert model.architecture.head_dimension == 128
    assert set(model.languages) == {"en", "fi", "zh"}
    assert ModelCapability.CODING in model.capabilities
    assert ModelCapability.REASONING in model.capabilities
    assert ModelCapability.TOOL_USE in model.capabilities
