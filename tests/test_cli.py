from typer.testing import CliRunner

from reachy_vec.cli import app
from reachy_vec.store.db import Store

from tests.conftest import FakeEmbedder

runner = CliRunner()


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
