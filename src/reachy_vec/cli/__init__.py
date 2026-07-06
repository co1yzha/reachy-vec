"""Command-line interface: reachy-vec {chat, ingest, enroll, run}."""

import typer

from reachy_vec.cli import chat, enroll, ingest, run, sync

app = typer.Typer(help="Team Familiar: embodied team assistant on Reachy Mini.")
app.command()(chat.chat)
app.command()(ingest.ingest)
app.command(name="sync-mongo")(sync.sync_mongo)
app.command()(enroll.enroll)
app.command()(run.run)


if __name__ == "__main__":
    app()
