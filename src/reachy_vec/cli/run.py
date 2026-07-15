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


def resolve_media_source(requested: str, media_available: bool) -> str:
    """Pick 'robot' or 'mac'. 'auto' -> robot when media is available, else mac."""
    if requested in ("robot", "mac"):
        return requested
    return "robot" if media_available else "mac"


def make_barge_in_factory(chosen: str, media):
    """Zero-arg factory building a fresh BargeInMonitor per reply, or None."""
    if not settings.barge_in:
        return None
    from reachy_vec.audio.listen import BargeInMonitor, MicSource
    from reachy_vec.audio.sources import RobotAudioSource

    def factory():
        source = (
            RobotAudioSource(media, target_rate=settings.audio_input_rate)
            if chosen == "robot"
            else MicSource()
        )
        return BargeInMonitor(source, min_speech_s=settings.barge_in_min_speech_s)

    return factory


def wrap_reconnect(body, connect_body, announce):
    """Wrap a RobotBody so motions survive a daemon/WiFi drop; pass others through."""
    from reachy_vec.body.robot import ReconnectingBody, RobotBody

    if settings.robot_reconnect and isinstance(body, RobotBody):
        return ReconnectingBody(
            connect_body=connect_body,
            max_attempts=settings.body_reconnect_attempts,
            announce=announce,
        )
    return body


def run(
    preview: bool = typer.Option(
        False, "--preview", help="Show a window with the webcam feed and face matches."
    ),
    source: str = typer.Option(
        None,
        "--source",
        help="Media source: auto | robot | mac (default: REACHY_VEC_MEDIA_SOURCE).",
    ),
) -> None:
    """Run the Oracle: face-triggered voice Q&A on the robot's or Mac's devices."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from reachy_vec.audio.listen import MicTranscriber, make_transcriber
    from reachy_vec.audio.sources import RobotAudioSource
    from reachy_vec.audio.speak import make_speaker
    from reachy_vec.body.robot import make_robot
    from reachy_vec.brain.chat import ChatBrain, default_opener
    from reachy_vec.brain.oracle import OracleLoop
    from reachy_vec.perception.camera import RobotCamera, WebcamCamera
    from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
    from reachy_vec.perception.vision import make_look_fn, make_selfie_fn
    from reachy_vec.perception.voice import EcapaSpeakerIdentifier
    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder

    load_dotenv()
    _setup_logging()
    store = Store(settings.lancedb_dir)
    if store.doc_count() == 0:
        typer.echo("Knowledge base is empty - run 'reachy-vec ingest <path>' first.")
        raise typer.Exit(code=1)

    # Connect the robot first (it decides whether robot media is available),
    # then build camera / mic source / speaker for the chosen world.
    requested = source or settings.media_source
    body, media = make_robot(with_media=requested in ("auto", "robot"))
    chosen = resolve_media_source(requested, media_available=media is not None)
    if chosen == "robot" and media is None:
        typer.echo(
            "--source robot but no robot media available (is the daemon up with "
            "media?).",
            err=True,
        )
        raise typer.Exit(code=1)

    if chosen == "robot":
        camera = RobotCamera(media)
        audio_source = RobotAudioSource(media, target_rate=settings.audio_input_rate)
    else:
        camera = WebcamCamera(settings.camera_index)
        audio_source = None  # MicSource default
    if camera.read() is None:
        typer.echo(f"No camera frame from '{chosen}' source - check the device.", err=True)
        raise typer.Exit(code=1)

    matcher = InsightFaceMatcher(store)
    speaker_id = EcapaSpeakerIdentifier(store)
    speaker = make_speaker(media=media if chosen == "robot" else None)
    embedder = BgeEmbedder(
        settings.embedding_model, query_prefix=settings.embedding_query_prefix
    )
    client = OpenAI()

    titles = store.demo_titles()
    vocab_prompt = f"Vocabulary: {', '.join(titles)}" if titles else None

    typer.echo("Warming up models (STT, faces, voices, embeddings)...")
    transcriber = make_transcriber(
        client=client, initial_prompt=vocab_prompt, source=audio_source
    )
    if isinstance(transcriber, MicTranscriber):
        transcriber.warm_up()
    matcher.observe(camera.read())   # loads insightface before the loop
    embedder.embed(["warm up"])      # loads the BGE model
    import numpy as np

    speaker_id.embed(np.zeros(32000, dtype=np.float32))  # loads ECAPA (2s of silence)
    try:                             # open the TLS connection to OpenAI
        client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,  # max_tokens is rejected by gpt-5* models
        )
    except Exception:
        pass  # no network now != no network later; the loop will surface it

    if preview:
        from reachy_vec.perception.preview import PreviewSight

        sight = PreviewSight(camera, matcher)
    else:
        sight = lambda: matcher.observe(camera.read())  # noqa: E731

    body = wrap_reconnect(
        body,
        connect_body=lambda: make_robot(with_media=False)[0],
        announce=speaker.speak,
    )
    look_fn = make_look_fn(
        camera,
        client,
        model=settings.vision_model or settings.llm_model,
        max_px=settings.vision_image_max_px,
        body=body,
    )
    selfie_fn = make_selfie_fn(
        camera,
        settings.photos_dir,
        body=body,
        speak=speaker.speak,
        opener=default_opener,
    )
    brain = ChatBrain(
        store=store,
        embedder=embedder,
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
        look_fn=look_fn,
        selfie_fn=selfie_fn,
    )
    loop = OracleLoop(
        sight=sight,
        transcriber=transcriber,
        speaker=speaker,
        body=body,
        brain=brain,
        enroll_capture=lambda name: enroll_person(
            name, camera, matcher, store, speaker.speak, faces_dir=settings.faces_dir
        ),
        store=store,
        greet_cooldown_s=settings.greet_cooldown_s,
        silence_timeout_s=settings.silence_timeout_s,
        idle_sleep_s=settings.idle_sleep_s,
        start_asleep=True,
        speaker_id=speaker_id,
        voice_passive_cap=settings.voice_passive_cap,
        barge_in_factory=make_barge_in_factory(chosen, media),
    )
    typer.echo("Oracle running - walk into frame. Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
