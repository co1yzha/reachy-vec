# Phase 1 "Oracle" — Design Spec

**Date:** 2026-07-06
**Project:** reachy-vec (Team Familiar)
**Parent spec:** [2026-07-06-team-familiar-design.md](2026-07-06-team-familiar-design.md)
**Status:** Approved design, pre-implementation

## 1. Goal

The walk-up demo, sim-first: a teammate appears in front of the (simulated)
robot, is recognized by face, gets a personalized spoken greeting with motion,
asks questions by voice, and hears answers from the Phase 0 RAG layer.
Unknown people are offered voice-driven enrollment.

**Milestone:** with the MuJoCo sim viewer open, `reachy-vec run` greets an
enrolled person by name (speech + motion), answers a spoken question about
ingested docs out loud, and enrolls a new person entirely by voice.

## 2. Decisions (locked during brainstorm)

1. **Conversation trigger: face-triggered.** Recognizing a face IS the wake
   signal — greet, then open the mic. No wake word, no push-to-talk (the
   Phase 0 terminal `chat` remains the debug fallback).
2. **Enrollment: robot-led.** On seeing a stable unknown face, Reachy offers
   enrollment by voice. Explicit spoken "yes" plus a name-confirmation round
   trip required; anything else → polite generic mode, no data stored.
3. **Greeting etiquette: per-person cooldown.** Full spoken greeting when the
   cooldown (default 2h, setting `REACHY_VEC_GREET_COOLDOWN_S`) has expired;
   otherwise a silent head-turn + antenna acknowledgment.
4. **Runtime architecture: threads + queues.** Camera thread and audio thread
   produce events; a synchronous state machine in the main thread consumes
   them. (asyncio rejected: every ML call is blocking anyway; multiprocess
   rejected: YAGNI.)
5. **Sim-only hardware mapping.** The sim's virtual camera sees an empty 3D
   scene, so the **Mac webcam is the eye**, Mac mic/speaker are ears/mouth,
   and the MuJoCo sim body is the body. Settings (`camera_source`,
   `audio_source`) flip to the robot's devices when hardware arrives; no code
   changes elsewhere.
6. **Face matching: insightface** (`buffalo_s` model pack), cosine similarity
   against stored embeddings. Conservative default threshold 0.45
   (setting `REACHY_VEC_FACE_THRESHOLD`); below threshold = unknown. Per the
   parent spec: never guess an identity.
7. **TTS: macOS `say` backend first** (already-designed pluggable backend);
   fish-speech voice clone is a follow-up swap inside `audio/speak.py`, not a
   redesign.

## 3. State machine (`reachy-vec run`)

```
IDLE ──known face──► GREET ──► LISTENING ──speech──► THINKING ──► SPEAKING ─┐
  ▲                                 ▲   │                                   │
  │                                 └───┴────── back to LISTENING ◄─────────┘
  └──── person gone / 30s silence ──┘
IDLE ──stable unknown face (~3s)──► OFFER_ENROLL ──"yes"──► ENROLLING ──► GREET
                                        │
                                        └─ anything else ──► generic mode (LISTENING, no identity)
```

- **IDLE:** face detection ~2 fps on the camera feed; body plays subtle idle
  motion. Known face ≥ threshold → GREET. Unknown face stably present ~3 s →
  OFFER_ENROLL.
- **GREET:** spoken "Hi <name>!" + greet motion if cooldown expired (update
  `last_greeted`); otherwise silent head-turn + antenna wiggle. → LISTENING.
- **LISTENING:** mic streaming; silero-VAD segments an utterance;
  faster-whisper transcribes. Robot holds listen pose. Utterance → THINKING.
- **THINKING:** Phase 0 `brain.rag.answer()` unchanged (docs RAG via OpenAI).
- **SPEAKING:** TTS through the speaker + gentle head motion. → LISTENING.
- **Exit to IDLE:** face absent from frame for ~5 s, or 30 s with no speech
  (setting `REACHY_VEC_SILENCE_TIMEOUT_S`) → goodbye nod → IDLE.
- **ENROLLING:** ask for name → STT → confirm ("Nice to meet you, <name> —
  did I get that right?") → on "yes", capture ~5 face frames while prompting
  (straight/left/right), store per-frame embeddings → GREET. On "no", retry
  name once, then give up politely.

## 4. Components

Fills the existing stubs; layout unchanged. All perception/audio units are
protocols with fakes, mirroring Phase 0's `Embedder` pattern.

| Module | Responsibility | Key tech |
|---|---|---|
| `body/robot.py` | daemon connection (`connection check`, head pose commands, graceful no-robot degradation) | reachy-mini SDK |
| `body/motions.py` | primitives: `greet`, `nod`, `listen_pose`, `idle`, `acknowledge`, `goodbye` | SDK goto's |
| `perception/face.py` | `FaceMatcher` protocol: detect faces in frame → embedding → best match `(person_id, score)` or unknown | insightface buffalo_s |
| `audio/listen.py` | `Transcriber` protocol: mic stream → VAD-segmented utterances → text | sounddevice, silero-vad, faster-whisper |
| `audio/speak.py` | `Speaker` protocol: text → audio out; backends `say` (now) / fish-speech (later) | subprocess `say` |
| `brain/loop.py` | `run_loop`: the state machine, consuming camera/audio events via queues (Phase 0 `chat_loop` stays for the terminal) | stdlib threading/queue |
| `store/schemas.py` + `db.py` | add `Person` rows: `person_id, name, face embeddings (one row per capture), last_greeted` + match/upsert helpers | lancedb |
| `cli/run.py` | wire real devices + state machine | typer |
| `cli/enroll.py` | launches the same robot-led enrollment flow directly (for pre-seeding without waiting to be noticed) | typer |

### People data model

`people` table, one row per captured face embedding (k-NN over all rows;
majority person wins):
`embedding_id, person_id, name, vector(512), created_at` — insightface
embeddings are 512-dim, distinct from the 384-dim text table.
Greeting state (`person_id, last_greeted`) lives in a small `greetings`
table updated on each spoken greeting.

## 5. Error handling

- **No daemon / robot unreachable:** `run` warns and continues voice+face
  only (motions become no-ops); body errors never kill the loop.
- **No webcam / no mic:** fail fast at startup with a clear message naming
  the missing device and the setting that selects it.
- **Borderline face match (within 0.05 of threshold):** treat as unknown but
  don't offer enrollment (avoids re-enrolling a bad angle of a known person).
- **STT empty/garbled during enrollment:** re-ask once, then abort politely.
- **OpenAI API failure in THINKING:** spoken apology ("Sorry, my brain isn't
  responding"), log, back to LISTENING.
- **Enrollment interrupted (person leaves):** discard partial captures.

## 6. Testing

- **Unit:** state machine driven entirely by fakes (`FakeFaceMatcher`,
  `FakeTranscriber`, `FakeSpeaker`, `FakeBody`) with scripted scenarios:
  known-person greet → question → answer → leave; unknown → enroll-yes;
  unknown → enroll-no; cooldown suppression; silence timeout.
- **Store:** people/greetings table round-trips with random 512-dim vectors.
- **Manual smoke per sub-plan:** sim viewer motion check; mic→STT echo test;
  webcam enrollment + self-recognition; full milestone run.
- Tests never load insightface/whisper models or open devices (Phase 0 rule:
  no network, no heavy models in unit tests).

## 7. Delivery plan

Four sub-plans, each independently shippable (mirrors 0a–0c):

1. **1a — Body:** `robot.py` + `motions.py`; smoke test = motions visible in
   sim viewer.
2. **1b — Voice:** `listen.py` + `speak.py`; smoke test = speak a sentence,
   have it transcribed back.
3. **1c — Faces:** `face.py` + people/greetings tables + basic `enroll` CLI
   capture; smoke test = enroll yourself, get recognized.
4. **1d — Oracle loop:** state machine + robot-led enrollment + `run` CLI;
   smoke test = the Phase 1 milestone.

1a/1b/1c are mutually independent (only 1d needs all three).

## 8. Out of scope (Phase 2+)

Voice/speaker identification and face+voice fusion; attributed memories
("note that…"); messages; fish-speech voice cloning (backend slot exists);
multi-person simultaneous conversations (first recognized face wins; others
acknowledged with a nod).
