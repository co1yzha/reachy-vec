import logging
from pathlib import Path

import typer

from reachy_vec.config import settings

logger = logging.getLogger(__name__)


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


def wait_for_frame(camera, timeout_s: float = 10.0, clock=None, sleep=None):
    """First frame from the camera, or None after timeout_s. The robot camera
    streams over WebRTC and takes a couple of seconds to deliver its first
    frame - a single instant read() would falsely report a dead camera."""
    import time

    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    deadline = clock() + timeout_s
    while True:
        frame = camera.read()
        if frame is not None or clock() >= deadline:
            return frame
        sleep(0.25)


def resolve_media_source(requested: str, media_available: bool) -> str:
    """Pick 'robot' or 'mac'. 'auto' -> robot when media is available, else mac."""
    if requested in ("robot", "mac"):
        return requested
    return "robot" if media_available else "mac"


def resolve_mic(media_source: str, mic_setting: str) -> str:
    """Whose mic to listen on: 'auto' follows the media source; 'mac' uses the
    Mac's default input (e.g. a USB mic) even with the robot speaker. The robot
    mic needs robot media, so it falls back to 'mac' without it."""
    if mic_setting == "mac" or media_source != "robot":
        return "mac"
    return "robot"


def _mac_has_input() -> bool:
    import sounddevice as sd

    try:
        return sd.query_devices(kind="input") is not None
    except Exception:  # PortAudioError "Error querying device -1": no input at all
        return False


def check_mic(mic: str, *, robot_media: bool, has_mac_input=_mac_has_input) -> str:
    """The Mac mic must exist (a USB mic can drop off); fall back to the robot
    mic when it doesn't, or stop with a clear message if there is none."""
    if mic != "mac" or has_mac_input():
        return mic
    if robot_media:
        typer.echo("No Mac microphone found (USB mic unplugged?) - using the robot's mic.",
                   err=True)
        return "robot"
    typer.echo("No microphone found - plug one in (System Settings > Sound > Input).",
               err=True)
    raise typer.Exit(code=1)


def make_barge_in_factory(chosen: str, media, is_self_voice=None, transcribe=None):
    """Zero-arg factory building a fresh BargeInMonitor per reply, or None.

    barge_in_mode=keyword (default): talking over a reply only interrupts when
    it says Reachy's name or a stop word ("Reachy, stop") - checked by
    transcribing the overlapping speech - so the mic hearing the robot's own
    voice can't cut replies short. speech: any sustained speech interrupts.
    """
    if not settings.barge_in:
        return None
    from reachy_vec.audio.listen import BargeInMonitor, MicSource
    from reachy_vec.audio.sources import RobotAudioSource
    from reachy_vec.brain.oracle import is_interrupt

    confirm = None
    if settings.barge_in_mode == "keyword":
        if transcribe is None:
            typer.echo("Barge-in by keyword needs a local transcriber - barge-in is off.",
                       err=True)
            return None

        def confirm(audio) -> bool:
            text = transcribe(audio)
            hit = is_interrupt(text, settings.wake_words)
            if hit:
                logger.info("barge-in phrase heard: %r", text)
            return hit

    def factory():
        source = (
            RobotAudioSource(media, target_rate=settings.audio_input_rate)
            if chosen == "robot"
            else MicSource()
        )
        return BargeInMonitor(
            source,
            min_speech_s=settings.barge_in_min_speech_s,
            is_self_voice=is_self_voice,
            confirm=confirm,
        )

    return factory


def _read_wav_16k(path):
    """Load a WAV as mono float32 at 16 kHz (the rate ECAPA embeds)."""
    import soundfile as sf

    from reachy_vec.audio.resample import to_mono_16k

    data, rate = sf.read(str(path), dtype="float32")
    return to_mono_16k(data, rate, 16000)


def make_self_voice_checker(store, identifier=None, read_sample=None):
    """is_self_voice(audio) for barge-in: does this sound like the robot?

    Only meaningful with a cloned TTS voice (qwen-tts): the ECAPA embedding
    of the barge-in audio is compared against the clone sample's. Returns
    None when there is no reference voice — the duration hysteresis is then
    the only guard. Caveat: a voice cloned from *you* also vetoes you.
    """
    sample = settings.voice_sample
    if settings.tts_backend != "qwen-tts" or sample is None or not Path(sample).is_file():
        return None
    read_sample = read_sample or _read_wav_16k
    state: dict = {}

    def is_self_voice(audio) -> bool:
        import numpy as np

        if "ident" not in state:
            from reachy_vec.perception.voice import EcapaSpeakerIdentifier

            # barge-in runs are ~0.7s; the default 1.0s minimum would always
            # return "can't tell", so this instance embeds shorter audio
            state["ident"] = identifier or EcapaSpeakerIdentifier(
                store, min_utterance_s=0.5
            )
            state["ref"] = state["ident"].embed(read_sample(sample))
        ref, vec = state["ref"], state["ident"].embed(audio)
        if ref is None or vec is None:
            return False  # can't tell -> behave like the pre-check monitor
        a, b = np.asarray(ref), np.asarray(vec)
        cosine = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
        return cosine >= settings.voice_threshold

    return is_self_voice


def prepare_fillers(speaker) -> list[str]:
    """Acknowledgement fillers for OracleLoop, pre-generated on the speaker
    (instant playback with the cloned voice); [] when disabled."""
    if not settings.ack_enabled or not settings.ack_fillers:
        return []
    speaker.precache(settings.ack_fillers)
    return list(settings.ack_fillers)


def make_camera_tools(media, *, client, body, speak, llm_kwargs=None, camera_cls=None,
                      frame_timeout_s: float = 10.0) -> dict:
    """ChatBrain kwargs for look / selfie / look_around on the robot camera, or {}
    when disabled, without robot media, or if no frame arrives (the Mac mini has
    no webcam)."""
    if not settings.camera_tools or media is None:
        return {}
    from reachy_vec.brain.chat import default_opener
    from reachy_vec.perception.camera import RobotCamera
    from reachy_vec.perception.vision import make_look_around_fn, make_look_fn, make_selfie_fn

    camera = (camera_cls or RobotCamera)(media)
    if wait_for_frame(camera, timeout_s=frame_timeout_s) is None:
        typer.echo("No frame from the robot camera - camera tools are off.", err=True)
        return {}
    model = settings.vision_model or settings.llm_model
    return {
        "look_fn": make_look_fn(camera, client, model=model, max_px=settings.vision_image_max_px,
                                body=body, llm_kwargs=llm_kwargs),
        "selfie_fn": make_selfie_fn(camera, settings.photos_dir, body=body, speak=speak,
                                    opener=default_opener),
        "look_around_fn": make_look_around_fn(
            camera, client, model=model, max_px=settings.vision_image_max_px, body=body,
            settle_s=settings.look_around_settle_s, llm_kwargs=llm_kwargs),
    }


def wrap_reconnect(body, connect_body, announce):
    """Wrap a RobotBody so motions survive a daemon/WiFi drop; pass others through."""
    from reachy_vec.body.robot import ReconnectingBody, RobotBody

    if settings.robot_reconnect and isinstance(body, RobotBody):
        return ReconnectingBody(
            connect_body=connect_body,
            max_attempts=settings.body_reconnect_attempts,
            announce=announce,
            initial=body,
        )
    return body


def stt_client(llm_client, settings=settings, openai_cls=None):
    """Client for make_transcriber: None for local STT (faster-whisper needs
    none, and building OpenAI() without OPENAI_API_KEY raises); for the
    openai backend, the cloud API - never the local LLM server."""
    from reachy_vec.brain.llm import uses_local_llm

    if settings.stt_backend != "openai":
        return None
    if not uses_local_llm(settings):
        return llm_client
    if openai_cls is None:
        from openai import OpenAI as openai_cls
    return openai_cls()


def make_hermes(store, settings=settings, which=None, gateway_cls=None,
                warm_in_background=True):
    """HermesJobs for the ask_hermes tool, or None when disabled/unavailable.

    hermes_transport=gateway talks to the running gateway's API server (no
    per-task startup, per-person memory); it falls back to the one-shot CLI
    when the key is missing, the gateway is down, or its API toolsets go
    beyond hermes_toolsets.
    """
    import shutil

    if not settings.hermes_enabled:
        return None
    from reachy_vec.brain.hermes import HermesCli, HermesError, HermesGateway, HermesJobs

    agent = None
    if settings.hermes_transport == "gateway":
        try:
            agent = (gateway_cls or HermesGateway)(
                base_url=settings.hermes_gateway_url,
                profile=settings.hermes_profile,
                api_key=settings.hermes_api_key,
            )
            agent.check_toolsets(allowed=set(settings.hermes_toolsets.split(",")))
        except (ValueError, HermesError) as e:
            typer.echo(f"Hermes gateway not usable ({e}) - using the hermes CLI.", err=True)
            agent = None
        else:
            if warm_in_background:
                import threading

                threading.Thread(target=agent.warm_up, daemon=True).start()
            else:
                agent.warm_up()
    if agent is None:
        which = which or shutil.which
        if which(settings.hermes_bin) is None:
            typer.echo(
                f"Hermes enabled but '{settings.hermes_bin}' not found on PATH - "
                "ask_hermes is off.",
                err=True,
            )
            return None
        # hermes -z loads AGENTS.md/rules from its CWD, so keep it out of the repo
        cwd = settings.data_dir / "hermes_cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        agent = HermesCli(
            bin=settings.hermes_bin,
            profile=settings.hermes_profile,
            toolsets=settings.hermes_toolsets,
            cwd=cwd,
        )
    return HermesJobs(
        agent,
        store,
        wait_s=settings.hermes_wait_s,
        wait_anonymous_s=settings.hermes_wait_anonymous_s,
        task_timeout_s=settings.hermes_task_timeout_s,
    )


def run(
    source: str = typer.Option(
        None,
        "--source",
        help="Media source: auto | robot | mac (default: REACHY_VEC_MEDIA_SOURCE).",
    ),
) -> None:
    """Run the Oracle: say "Reachy" to start a voice Q&A; people are known by voice."""
    from dotenv import load_dotenv

    from reachy_vec.audio.listen import make_transcriber
    from reachy_vec.audio.sources import RobotAudioSource
    from reachy_vec.audio.speak import make_speaker
    from reachy_vec.audio.vocab import VocabSnappingTranscriber
    from reachy_vec.body.robot import make_robot
    from reachy_vec.brain.chat import ChatBrain
    from reachy_vec.brain.llm import make_llm_client, uses_local_llm
    from reachy_vec.brain.oracle import OracleLoop
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
    # then build mic source / speaker for the chosen world. Vision (camera,
    # look/selfie/look_around tools, camera-poke startle) is disabled for now.
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

    mic = check_mic(resolve_mic(chosen, settings.mic_source), robot_media=media is not None)
    if mic == "robot":
        audio_source = RobotAudioSource(media, target_rate=settings.audio_input_rate)
    else:
        audio_source = None  # MicSource default

    speaker_id = EcapaSpeakerIdentifier(store)
    speaker = make_speaker(media=media if chosen == "robot" else None)
    embedder = BgeEmbedder(
        settings.embedding_model, query_prefix=settings.embedding_query_prefix
    )
    client = make_llm_client()

    # Demo titles are matched *after* transcription (audio/vocab.py) — feeding
    # them to STT as a prompt made it hallucinate titles verbatim on echo/noise.
    titles = store.demo_titles()

    typer.echo("Warming up models (STT, voices, embeddings)...")
    transcriber = make_transcriber(client=stt_client(client), source=audio_source)
    # barge-in phrase check reuses the loaded local STT model (None for cloud STT)
    interrupt_stt = getattr(transcriber, "transcribe_audio", None)
    if hasattr(transcriber, "warm_up"):  # local models: load before the first utterance
        transcriber.warm_up()
    if titles:
        transcriber = VocabSnappingTranscriber(transcriber, titles)
    embedder.embed(["warm up"])      # loads the BGE model
    import numpy as np

    speaker_id.embed(np.zeros(32000, dtype=np.float32))  # loads ECAPA (2s of silence)

    body = wrap_reconnect(
        body,
        connect_body=lambda: make_robot(with_media=False)[0],
        announce=speaker.speak,
    )
    if chosen != "robot":
        # Robot-speaker replies get the SDK's audio-synced wobble (make_robot);
        # Mac-speaker replies get a gentle sway loop instead.
        from reachy_vec.body.sway import SpeakingSway, SwayingSpeaker

        speaker = SwayingSpeaker(speaker, SpeakingSway(body))
    hermes = make_hermes(store)
    thinking_off = uses_local_llm() and settings.llm_disable_thinking
    camera_tools = make_camera_tools(
        media, client=client, body=body, speak=speaker.speak,
        llm_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        if thinking_off else None,
    )
    brain = ChatBrain(
        k=settings.rag_k,
        rag_min_score=settings.rag_min_score,
        store=store,
        embedder=embedder,
        client=client,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        web_search=settings.web_search,
        disable_thinking=thinking_off,
        hermes=hermes,
        fix_body_fn=getattr(body, "fix", None),  # "get your head up": motors on + wake
        **camera_tools,
    )
    if not brain.warm_up():
        typer.echo(
            f"LLM not reachable at {settings.llm_base_url or 'OpenAI'} - "
            "answers will fail until it is.",
            err=True,
        )
    fillers = prepare_fillers(speaker)
    loop = OracleLoop(
        transcriber=transcriber,
        speaker=speaker,
        body=body,
        brain=brain,
        store=store,
        speaker_id=speaker_id,
        wake_words=settings.wake_words,
        greet_cooldown_s=settings.greet_cooldown_s,
        silence_timeout_s=settings.silence_timeout_s,
        wake_listen_s=settings.wake_listen_s,
        idle_sleep_s=settings.idle_sleep_s,
        start_asleep=True,
        voice_threshold=settings.voice_threshold,
        voice_passive_cap=settings.voice_passive_cap,
        enroll_phrases=settings.enroll_voice_phrases,
        barge_in_factory=make_barge_in_factory(
            mic, media, is_self_voice=make_self_voice_checker(store),
            transcribe=interrupt_stt,
        ),
        settle_s=settings.post_speech_settle_s,
        ack_fillers=fillers,
        name_extractor=brain.extract_name,  # LLM reads the name from the answer
        auto_offer=settings.enroll_auto_offer,
    )
    typer.echo('Oracle running - say "Reachy" to start. Ctrl+C to stop.')
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
    finally:
        if hermes is not None:
            hermes.close()
