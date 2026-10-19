"""Import a downloaded Hugging Face metadata snapshot without network access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lumi_fit.domain.models import (
    ArtifactFormat,
    ArtifactRecord,
    ModelCapability,
    ModelRecord,
    TransformerArchitecture,
    WeightDtype,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Hugging Face config requires positive integer {key!r}")
    return value


def _weight_dtype(torch_dtype: str) -> tuple[WeightDtype, str]:
    normalized = torch_dtype.lower()
    mapping = {
        "bfloat16": (WeightDtype.BF16, "bf16"),
        "float16": (WeightDtype.FP16, "fp16"),
        "float32": (WeightDtype.FP32, "fp32"),
    }
    return mapping.get(normalized, (WeightDtype.OTHER, normalized))


def _capabilities(api: dict[str, Any]) -> tuple[ModelCapability, ...]:
    tags = {str(tag).lower() for tag in api.get("tags", [])}
    capabilities: list[ModelCapability] = []
    if api.get("pipeline_tag") == "text-generation" or "text-generation" in tags:
        capabilities.append(ModelCapability.TEXT_GENERATION)
    if "conversational" in tags:
        capabilities.append(ModelCapability.CHAT)
    if "code" in tags or "coding" in tags:
        capabilities.append(ModelCapability.CODING)
    if "reasoning" in tags:
        capabilities.append(ModelCapability.REASONING)
    if "function-calling" in tags or "tool-use" in tags:
        capabilities.append(ModelCapability.TOOL_USE)
    if "feature-extraction" in tags or api.get("pipeline_tag") == "feature-extraction":
        capabilities.append(ModelCapability.EMBEDDING)
    return tuple(capabilities or [ModelCapability.TEXT_GENERATION])


def _languages(api: dict[str, Any]) -> tuple[str, ...]:
    card_data = api.get("cardData")
    declared = card_data.get("language", ()) if isinstance(card_data, dict) else ()
    if isinstance(declared, str):
        declared = (declared,)
    if not isinstance(declared, (list, tuple)):
        declared = ()
    tag_languages = (
        str(tag).lower()
        for tag in api.get("tags", [])
        if re.fullmatch(r"[a-z]{2,3}", str(tag).lower())
    )
    return tuple(sorted({str(language).lower() for language in declared} | set(tag_languages)))


def _license(api: dict[str, Any]) -> str | None:
    for tag in api.get("tags", []):
        text = str(tag)
        if text.startswith("license:"):
            return text.removeprefix("license:")
    return None


def _artifact_id(model_id: str, quantization: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{slug}-{quantization}-safetensors"


def import_huggingface_snapshot(
    *,
    config_path: str | Path,
    api_path: str | Path,
    safetensors_index_path: str | Path,
    compatible_runtime_ids: tuple[str, ...] = ("vllm-laif",),
) -> tuple[ModelRecord, ArtifactRecord]:
    """Normalize official HF config/API/index files into catalog records.

    Acquisition is intentionally separate. The importer accepts local files so
    catalog refreshes can be reviewed, reproduced, and tested without hidden
    network behavior.
    """

    config = _read_json(config_path)
    api = _read_json(api_path)
    index = _read_json(safetensors_index_path)

    model_id = api.get("id")
    revision = api.get("sha")
    if not isinstance(model_id, str) or "/" not in model_id:
        raise ValueError("Hugging Face API record requires a namespaced model id")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Hugging Face API record requires an immutable revision SHA")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("Hugging Face config requires at least one architecture")

    safetensors = api.get("safetensors")
    metadata = index.get("metadata")
    if not isinstance(safetensors, dict) or not isinstance(metadata, dict):
        raise ValueError("Hugging Face metadata requires safetensors size information")
    total_parameters = safetensors.get("total")
    total_bytes = metadata.get("total_size")
    if not isinstance(total_parameters, int) or total_parameters < 1:
        raise ValueError("Hugging Face API record requires safetensors.total")
    if not isinstance(total_bytes, int) or total_bytes < 1:
        raise ValueError("safetensors index requires metadata.total_size")

    dtype_name = config.get("torch_dtype", config.get("dtype", "unknown"))
    dtype, quantization = _weight_dtype(str(dtype_name))
    created_at = api.get("createdAt")
    release_date = created_at[:10] if isinstance(created_at, str) else None
    family = str(config.get("model_type", architectures[0]))

    hidden_size = _required_int(config, "hidden_size")
    attention_heads = _required_int(config, "num_attention_heads")
    head_dimension = config.get("head_dim")
    if head_dimension is None:
        if hidden_size % attention_heads:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when head_dim is absent"
            )
        head_dimension = hidden_size // attention_heads

    architecture = TransformerArchitecture(
        architecture_name=str(architectures[0]),
        total_parameters=total_parameters,
        active_parameters=config.get("active_parameters"),
        layers=_required_int(config, "num_hidden_layers"),
        hidden_size=hidden_size,
        intermediate_size=_required_int(config, "intermediate_size"),
        attention_heads=attention_heads,
        kv_heads=_required_int(config, "num_key_value_heads"),
        head_dimension=head_dimension,
        vocabulary_size=_required_int(config, "vocab_size"),
        context_length=_required_int(config, "max_position_embeddings"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        moe_experts=config.get("num_experts"),
        active_experts_per_token=config.get("num_experts_per_tok"),
    )
    source = f"https://huggingface.co/{model_id}"
    model = ModelRecord(
        model_id=model_id,
        revision=revision,
        family=family,
        release_date=release_date,
        license=_license(api),
        gated=bool(api.get("gated", False)),
        capabilities=_capabilities(api),
        languages=_languages(api),
        architecture=architecture,
        source_url=source,
    )
    artifact = ArtifactRecord(
        artifact_id=_artifact_id(model_id, quantization),
        model_id=model_id,
        revision=revision,
        format=ArtifactFormat.SAFETENSORS,
        weight_dtype=dtype,
        quantization=quantization,
        total_bytes=total_bytes,
        compatible_runtime_ids=compatible_runtime_ids,
        source_url=f"{source}/blob/{revision}/model.safetensors.index.json",
    )
    return model, artifact
