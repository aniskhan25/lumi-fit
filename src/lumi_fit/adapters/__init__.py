"""Side-effecting adapters that normalize external data for the pure core."""

from .hf_import import import_huggingface_snapshot

__all__ = ["import_huggingface_snapshot"]
