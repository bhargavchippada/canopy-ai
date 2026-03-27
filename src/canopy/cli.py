"""Canopy CLI — placeholder for future commands."""

import typer

app = typer.Typer(help="Canopy: Evolving decision trees for behavioral profiling.")


@app.command()
def version() -> None:
    """Show version."""
    from canopy import __version__
    typer.echo(f"canopy-ai v{__version__}")
