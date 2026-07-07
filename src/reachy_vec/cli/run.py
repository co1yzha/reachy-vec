import logging

import typer

from reachy_vec.config import settings


def _setup_logging() -> None:
    """INFO-level reachy_vec logs (heard utterances, opened URLs, errors)
    to console and data/reachy.log. Transcripts of everyone who talks to
    the robot end up in this file - delete it to forget."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(settings.data_dir / "reachy.log"),
    ]
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    app_logger = logging.getLogger("reachy_vec")
    app_logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)


def run(
    preview: bool = typer.Option(
        False, "--preview", help="Show a window with the webcam feed and face matches."
    ),
) -> None:
    """Run the Oracle: face-triggered voice Q&A on webcam + mic (+ sim body)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from reachy_vec.audio.listen import MicTranscriber, make_transcriber
    from reachy_vec.audio.speak import make_speaker
    from reachy_vec.body.robot import make_body
    from reachy_vec.brain.chat import ChatBrain
    from reachy_vec.brain.oracle import OracleLoop
    from reachy_vec.perception.camera import WebcamCamera
    from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder

    load_dotenv()
    _setup_logging()
    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)

    camera = WebcamCamera(settings.camera_index)
    if camera.read() is None:
        typer.echo("No camera frame - check webcam permission/index.", err=True)
        raise typer.Exit(code=1)

    matcher = InsightFaceMatcher(store)
    speaker = make_speaker()
    embedder = BgeEmbedder(settings.embedding_model)
    client = OpenAI()

    titles = store.demo_titles()
    vocab_prompt = f"Vocabulary: {', '.join(titles)}" if titles else None

    typer.echo("Warming up models (STT, faces, embeddings)...")
    transcriber = make_transcriber(client=client, initial_prompt=vocab_prompt)
    if isinstance(transcriber, MicTranscriber):
        transcriber.warm_up()
    matcher.observe(camera.read())   # loads insightface before the loop
    embedder.embed(["warm up"])      # loads the BGE model

    if preview:
        from reachy_vec.perception.preview import PreviewSight

        sight = PreviewSight(camera, matcher)
    else:
        sight = lambda: matcher.observe(camera.read())  # noqa: E731

    brain = ChatBrain(
        store=store, embedder=embedder, client=client, model=settings.llm_model
    )
    loop = OracleLoop(
        sight=sight,
        transcriber=transcriber,
        speaker=speaker,
        body=make_body(),
        brain=brain,
        enroll_capture=lambda name: enroll_person(
            name, camera, matcher, store, speaker.speak, faces_dir=settings.faces_dir
        ),
        store=store,
        greet_cooldown_s=settings.greet_cooldown_s,
        silence_timeout_s=settings.silence_timeout_s,
        idle_sleep_s=settings.idle_sleep_s,
    )
    typer.echo("Oracle running - walk into frame. Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
