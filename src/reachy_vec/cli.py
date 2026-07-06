"""Entry points: reachy-vec {chat, ingest, enroll, run}."""

import typer

app = typer.Typer(help="Team Familiar: embodied team assistant on Reachy Mini.")


@app.command()
def chat() -> None:
    """Phase 0: text-only RAG loop in the terminal (no robot needed)."""
    raise typer.Exit("Not implemented yet (Phase 0).")


@app.command()
def ingest(path: str) -> None:
    """Phase 0: ingest team documents at PATH into the knowledge base."""
    raise typer.Exit("Not implemented yet (Phase 0).")


@app.command()
def enroll(name: str) -> None:
    """Phase 1: enroll a teammate — capture face and voice samples."""
    raise typer.Exit("Not implemented yet (Phase 1).")


@app.command()
def run() -> None:
    """Phase 1: full loop — connect to the robot and start the assistant."""
    raise typer.Exit("Not implemented yet (Phase 1).")


if __name__ == "__main__":
    app()
