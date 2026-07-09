from typer.testing import CliRunner

from reachy_vec.cli import app
from reachy_vec.cli.run import resolve_media_source
from reachy_vec.store.db import Store
from tests.conftest import FakeEmbedder

runner = CliRunner()


def test_resolve_auto_prefers_robot_when_available():
    assert resolve_media_source("auto", media_available=True) == "robot"


def test_resolve_auto_falls_back_to_mac():
    assert resolve_media_source("auto", media_available=False) == "mac"


def test_resolve_explicit_robot_is_honored_even_if_unavailable():
    assert resolve_media_source("robot", media_available=False) == "robot"


def test_resolve_explicit_mac():
    assert resolve_media_source("mac", media_available=True) == "mac"


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "ingest", "enroll", "run"):
        assert command in result.output


def test_ingest_command_writes_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha content")
    db_dir = tmp_path / "lancedb"

    monkeypatch.setattr("reachy_vec.cli.ingest.make_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("reachy_vec.cli.ingest.settings.data_dir", tmp_path)

    result = runner.invoke(app, ["ingest", str(docs)])
    assert result.exit_code == 0, result.output
    assert "1 chunk" in result.output
    assert Store(db_dir).doc_count() == 1


def test_ingest_command_rejects_missing_path(tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope")])
    assert result.exit_code != 0
