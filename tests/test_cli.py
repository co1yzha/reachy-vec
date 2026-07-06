from typer.testing import CliRunner

from reachy_vec.cli import app

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "ingest", "enroll", "run"):
        assert command in result.output
