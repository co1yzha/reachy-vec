import time

import typer

from reachy_vec.config import settings
from reachy_vec.perception.camera import WebcamCamera
from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
from reachy_vec.store.db import Store


def enroll(name: str) -> None:
    """Enroll a teammate's face from the webcam (keyboard-guided)."""
    store = Store(settings.lancedb_dir)
    camera = WebcamCamera(settings.camera_index)
    matcher = InsightFaceMatcher(store)

    def prompt(msg: str) -> None:
        typer.echo(f">> {msg}")
        time.sleep(1.5)  # give the person a beat to move

    person_id = enroll_person(name, camera, matcher, store, prompt)
    if person_id is None:
        typer.echo("No usable face captured - check lighting/camera and retry.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Enrolled {name} ({person_id}) with face data.")
