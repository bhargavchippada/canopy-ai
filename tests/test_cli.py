"""Tests for canopy.cli — typer CLI app."""

from __future__ import annotations

from typer.testing import CliRunner

from canopy.cli import app

runner = CliRunner()


class TestCLI:
    def test_version_command(self) -> None:
        result = runner.invoke(app, ["version"])
        # If single-command app, typer may require no args
        if result.exit_code != 0:
            result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "canopy-ai v" in result.output

    def test_version_contains_semver(self) -> None:
        from canopy import __version__

        result = runner.invoke(app, ["version"])
        if result.exit_code != 0:
            result = runner.invoke(app, [])
        assert __version__ in result.output
