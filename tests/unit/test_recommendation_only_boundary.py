import ast
from pathlib import Path

from lumi_fit.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]


def test_source_does_not_import_process_or_remote_execution_modules() -> None:
    forbidden = {"subprocess", "paramiko", "fabric", "asyncssh"}
    imported: set[str] = set()
    for path in (ROOT / "src/lumi_fit").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(forbidden)


def test_cli_exposes_recommendation_commands_only() -> None:
    help_text = build_parser().format_help().casefold()

    assert "models" in help_text
    assert "plan" in help_text
    assert "compare" in help_text
    assert "explain" in help_text
    for forbidden in ("deploy", "render", "submit", "launch", "probe", "benchmark"):
        assert forbidden not in help_text
