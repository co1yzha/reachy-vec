import typer

from reachy_vec.config import settings


def run() -> None:
    """Run the Oracle: face-triggered voice Q&A on webcam + mic (+ sim body)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from reachy_vec.audio.listen import MicTranscriber, make_transcriber
    from reachy_vec.audio.speak import make_speaker
    from reachy_vec.body.robot import make_body
    from reachy_vec.brain.oracle import OracleLoop
    from reachy_vec.brain.rag import answer
    from reachy_vec.perception.camera import WebcamCamera
    from reachy_vec.perception.face import InsightFaceMatcher, enroll_person
    from reachy_vec.store.db import Store
    from reachy_vec.store.embeddings import BgeEmbedder

    load_dotenv()
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

    loop = OracleLoop(
        sight=lambda: matcher.observe(camera.read()),
        transcriber=transcriber,
        speaker=speaker,
        body=make_body(),
        answer_fn=lambda q: answer(
            q, store=store, embedder=embedder, client=client, model=settings.llm_model
        ),
        enroll_capture=lambda name: enroll_person(
            name, camera, matcher, store, speaker.speak
        ),
        store=store,
        greet_cooldown_s=settings.greet_cooldown_s,
        silence_timeout_s=settings.silence_timeout_s,
    )
    typer.echo("Oracle running - walk into frame. Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
