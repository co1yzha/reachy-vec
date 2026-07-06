# Team Familiar — Design Spec

**Date:** 2026-07-06
**Project:** reachy-vec
**Status:** Approved concept, pre-implementation

## 1. Aims

Build a team-facing embodied assistant on a **Reachy Mini (wireless, Raspberry Pi 5)** whose brain runs on a **Mac over WiFi**. The robot sits in a shared space and:

1. **Knows who it's talking to.** Recognizes enrolled teammates by face (camera) and confirms by voice (speaker embeddings). Attributes every utterance to the right person, even with multiple people present.
2. **Answers from a shared team knowledge base.** Voice Q&A via retrieval-augmented generation over team docs and notes, personalized to the recognized speaker.
3. **Remembers per person.** Captures spoken notes ("Reachy, note that the pipeline is fixed"), attributes and stores them, and answers questions like "what did Alice say about the dataset?"
4. **Relays messages between teammates.** "Tell Bob the meeting moved" → delivered when it next recognizes Bob.
5. **Is expressive.** Uses head pose, body rotation, and antennas to greet, listen, and react — the embodiment is part of the product, not decoration.

**Non-goals (for now):** OpenClaw/NemoClaw integration, on-robot inference, cloud deployment, mobile access, guardrail/policy frameworks. These are possible later phases, not foundations.

## 2. Architecture

```
┌─────────────────────┐   WiFi    ┌──────────────────────────────┐
│  Reachy Mini        │◄─────────►│  Brain (Mac, Apple silicon)  │
│  (wireless, Pi 5)   │  SDK/API  │                              │
│  camera, 4-mic,     │           │  Perception: face ID         │
│  speaker, motion    │           │   (insightface) + speaker ID │
└─────────────────────┘           │   (ECAPA voice embeddings)   │
                                  │  STT: faster-whisper (local) │
                                  │  LLM: OpenAI API + RAG       │
                                  │  TTS: fish-speech (cloned)   │
                                  │  Store: LanceDB (embedded)   │
                                  └──────────────────────────────┘
```

- **Robot = thin body.** Streams audio/video to the Mac; receives speech audio and motion commands via the Reachy Mini Python SDK / REST API.
- **Mac = brain.** All heavy models run locally on Apple silicon; the LLM is Claude via API.
- **One store for all vectors.** LanceDB (embedded, no server) holds: document chunks, conversation/notes memory, face embeddings, and voice embeddings.

### Components

Each component is a subpackage under `src/reachy_vec/`:

| Component | Responsibility | Key tech |
|---|---|---|
| `body/` | Robot I/O: connection/streaming (`robot.py`), motion primitives — greet, nod, track, droop (`motions.py`) | reachy-mini SDK |
| `perception/` | Face recognition (`face.py`), speaker ID (`voice.py`), identity fusion (`fusion.py`) | insightface, speechbrain ECAPA |
| `audio/` | VAD + wake word + streaming STT (`listen.py`), TTS to robot speaker (`speak.py`) | faster-whisper, silero-vad, fish-speech |
| `brain/` | Intent routing (`intents.py`), RAG prompting (`rag.py`), conversation loop (`loop.py`) | OpenAI API |
| `store/` | LanceDB connection (`db.py`) and table schemas (`schemas.py`): `people`, `docs`, `memories`, `messages` | lancedb |
| `cli/` | One file per command: `chat`, `ingest`, `enroll`, `run` | typer |

### Data model (LanceDB tables)

- **people**: `person_id, name, face_embeddings[], voice_embeddings[], preferences`
- **docs**: `chunk_id, text, embedding, source, ingested_at`
- **memories**: `memory_id, person_id (speaker), text, embedding, timestamp`
- **messages**: `message_id, from_person, to_person, text, created_at, delivered_at?`

### Error handling principles

- Unknown face/voice → polite generic mode, offer enrollment; never guess an identity.
- Face and voice ID disagree → trust neither; ask ("Sorry — who am I speaking with?").
- Robot unreachable over WiFi → brain degrades to headless CLI mode for testing.
- STT low confidence → ask to repeat rather than act on a garbled note/message.

## 3. Roadmap

### Phase 0 — Skeleton & simulator (no hardware required)
- Project scaffold, LanceDB store module, config.
- Text-only brain loop: type a question → RAG answer from ingested docs.
- `ingest` CLI working end-to-end.
- **Milestone:** correct RAG answers over a sample team doc set, in the terminal.

### Phase 1 — Oracle (the walk-up demo)
- Connect to the wireless Reachy Mini over WiFi; motion primitives (greet, idle, listen pose).
- `enroll` CLI: face + voice enrollment.
- Face recognition on camera stream → personalized spoken greeting with head-turn.
- Wake word → STT → RAG → TTS voice answer.
- **Milestone:** teammate walks up, is greeted by name, asks a question, gets a spoken answer.

### Phase 2 — Memory keeper (attribution)
- Speaker ID fused with face ID per utterance.
- "Note that…" intent → attributed memory stored in LanceDB.
- Queries over memories: "what did Alice say about X?"
- **Milestone:** two people take turns speaking; utterances are attributed correctly and recallable.

### Phase 3 — Messenger
- "Tell Bob…" intent → stored message.
- On recognizing Bob: deliver pending messages, mark delivered.
- **Milestone:** async message relayed across two sightings.

### Phase 4 (optional, later) — Hardening
- Guardrails on responses, audit log of who-asked-what (NemoClaw-style policy if ever needed), on-robot fallback modes.

## 4. Key decisions & rationale

- **Custom stack over OpenClaw/ClawBody/NemoClaw:** multi-user identity and team KB are first-class here and absent there; NemoClaw solves deployment safety, not perception or attribution, and is NVIDIA-GPU oriented.
- **LanceDB over MongoDB:** embedded, zero-ops, fast on-device vector search; one store for docs, memories, and biometric embeddings. MongoDB not required by the user.
- **Mac as brain:** wireless Reachy's Pi 5 can't run whisper + face ID + embeddings in real time; Apple silicon can, and dev happens on macOS anyway.
- **OpenAI as LLM:** user has an OpenAI API key; the RAG layer is provider-agnostic enough to swap later.
- **TTS = fish-speech first, openvoice fallback:** fish-speech (OpenAudio S1-mini) has the best voice-clone quality and active development; its risk is latency on Apple silicon MPS, mitigated by short spoken responses. OpenVoice v2 is lighter/faster but lower quality and dormant. Backend is pluggable via `settings.tts_backend` (`fish-speech` / `openvoice` / macOS `say` for dev), with `settings.voice_sample` as the cloning reference.
- **Consent note:** face/voice enrollment is explicit and opt-in per teammate; embeddings stay local in LanceDB.
