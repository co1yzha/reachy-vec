"""CLI: reachy-vec {chat, ingest, sync-mongo, run, dashboard, doctor, record-voice}."""

import typer

from reachy_vec.cli import chat, dashboard, doctor, ingest, record_voice, run, sync

app = typer.Typer(help="Team Familiar: embodied team assistant on Reachy Mini.")
app.command()(chat.chat)
app.command()(ingest.ingest)
app.command(name="sync-mongo")(sync.sync_mongo)
app.command()(run.run)
app.command()(dashboard.dashboard)
app.command()(doctor.doctor)
app.command(name="record-voice")(record_voice.record_voice)


if __name__ == "__main__":
    app()
