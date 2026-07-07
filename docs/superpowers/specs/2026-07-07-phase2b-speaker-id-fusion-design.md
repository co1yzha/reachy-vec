# Phase 2b — Speaker ID + identity fusion

**Date:** 2026-07-07 · **Status:** approved in conversation
**Parent:** [Team Familiar spec](2026-07-06-team-familiar-design.md) (delivers
the deferred voice half of Phase 2; cross-person recall — "what did Alice
say about X?" — remains out of scope).

## Goal

Attribute every utterance to the right person by voice, fused with face ID,
so that with two people in frame the notes, memories, and messages go to
whoever actually spoke — and an unknown voice is answered politely but never
written to the store. Completes the original Phase 2 milestone: "two people
take turns speaking; utterances are attributed correctly and recallable."

## Decisions

1. **Per-utterance attribution.** Voice ID runs on every utterance, not
   just once at greeting. The conversation continues as one visit, but each
   turn's identity follows the voice.
2. **Explicit enrollment + passive backfill.** New enrollments add one
   spoken-phrase capture after the five face captures. Existing enrollees
   need no re-enrollment: confident solo face matches passively bank
   utterance embeddings into their voice profile.
3. **Follow the voice per turn.** A different enrolled voice mid-conversation
   is recognized and served, not interrogated. Unknown voice → answer
   normally, refuse side-effects (save_note / send_message).
4. **Audio comes from the transcriber (approach A).** `listen_once` already
   captures the VAD-segmented utterance before STT; it now returns the audio
   alongside the text. One mic capture, no second tap, no coupling of STT
   with identity.

## Architecture

```
mic ──► Transcriber.listen_once ──► Utterance(text, audio 16kHz float32)
                                          │
             face Observation (poll)      ▼
                        └──────► fuse(face_obs, voice_obs) ──► TurnIdentity
                                          ▲                        │
        EcapaSpeakerIdentifier.identify(audio) ── voices table     ▼
                                              ChatBrain.respond(text, identity)
```

### `audio/listen.py` — Utterance

- New frozen dataclass `Utterance(text: str, audio: np.ndarray | None)`.
- `Transcriber.listen_once(timeout_s) -> Utterance | None` (was `str | None`).
  Both backends return the captured mono 16 kHz float32 array they already
  have in hand; `None` still means silence/timeout.
- All call sites update: the Oracle conversation loop, enrollment yes/no and
  name prompts (which just use `.text`), and the fakes in `tests/conftest.py`.

### `perception/voice.py` — speaker ID

- `SpeakerIdentifier` protocol: `identify(audio) -> Observation | None`
  (reuses the `Observation` dataclass from `face.py`: person_id, name,
  score; `None` = not enough signal). `embed(audio) -> list[float] | None`.
- `EcapaSpeakerIdentifier`: speechbrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`),
  192-dim embeddings, lazy-loaded on first use — same shape as
  `InsightFaceMatcher`. Utterances shorter than ~1 s return `None` (too
  little signal to judge).
- Matching mirrors faces: k-NN (k=5, cosine) majority vote over the `voices`
  table via `Store.match_voice`, gated by `VOICE_THRESHOLD` with the same
  borderline-margin rule (within the margin under threshold → `None`, i.e.
  "can't tell", not "unknown").
- speechbrain moves from the `perception` optional extra into main
  dependencies (it is now a runtime requirement of `run`).

### `store` — voices table

New LanceDB table `voices` (`VoiceRow`):

| field | type | notes |
|---|---|---|
| voice_id | str | `{person_id}:{uuid}` |
| person_id | str | FK to people rows |
| name | str | denormalized, as in FaceRow |
| vector | 192 floats | ECAPA embedding |
| created_at | str | ISO UTC |
| source | str | `enrolled` \| `passive` |

`Store` gains: `add_voice_rows`, `match_voice` (same vote as `match_face`),
`voice_row_count(person_id)`, `prune_passive_voices(person_id, keep)`.

### `perception/fusion.py` — pure fusion function

`fuse(face_obs, voice_obs) -> TurnIdentity` where `TurnIdentity` is
`(person_id | None, name | None)`. Truth table ("never guess", per turn):

| voice says | face says | turn identity |
|---|---|---|
| known P | anything | P (voice wins — the speaker may be off-camera) |
| unknown (below threshold) | anything | anonymous |
| can't tell (None / too short / borderline) | known Q | Q |
| can't tell | unknown / no face | anonymous |

The face observation used is the most recent one the loop already has (the
sighting that started or last refreshed the conversation) — no extra camera
poll per turn. Face is the tie-breaker, voice is the authority. This
supersedes the parent spec's "face and voice disagree → ask" rule: asking
interrupts real multi-person conversations; following a *confident* voice
match is still never-guessing, because anything below threshold stays
anonymous. Pure function → exhaustive table-driven tests.

### `brain/chat.py` — per-turn identity

- `respond(question, identity: TurnIdentity, on_sentence=...)` replaces the
  `speaker_name` parameter. Per turn:
  - memory retrieval searches the *turn speaker's* memories;
  - `save_note` and `send_message` attribute to the turn speaker;
  - anonymous turn → retrieval skips memories, side-effect tools return
    "I don't know who's asking" (existing behavior, now per turn);
  - the history line is labeled with the turn speaker's name (template
    already supports this).
- Terminal `chat` (no camera/mic) passes an anonymous identity, preserving
  today's docs-only behavior.
- `begin_conversation(person_id, name)` stays: it sets the *visit owner*
  used for greeting continuity and as the summary subject. End-of-visit
  distillation groups the transcript by speaker label and stores up to 3
  notes per enrolled person who spoke (anonymous turns are never distilled).

### `brain/oracle.py` — loop changes

Per conversation turn: `utterance = transcriber.listen_once(...)` →
`voice_obs = speaker_id.identify(utterance.audio)` → `identity =
fuse(last_face_obs, voice_obs)` → `brain.respond(utterance.text, identity,
on_sentence=...)`. Silence handling, greeting, cooldown, message delivery,
and sleep/wake are unchanged.

### Enrollment + passive backfill

- **Explicit:** after the five face captures, the robot says "Now say a
  sentence so I learn your voice — anything you like." One utterance
  (`listen_once(10)`), embedded, stored as one `enrolled` voice row. No
  usable audio → enrollment still succeeds face-only (voice arrives later
  via backfill); the robot says it will learn the voice as they talk.
- **Passive backfill:** after a turn where the face is a confident solo
  match for person P (score ≥ face threshold, exactly one face) **and** the
  voice observation is not a different known person, the utterance embedding
  is stored as a `passive` row for P. Cap: 10 passive rows per person,
  oldest evicted (`prune_passive_voices`). Utterances < 1 s are skipped.

## Configuration (new, in `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `VOICE_THRESHOLD` | `0.30` | cosine gate for voice recognition (ECAPA scores run lower than face; tune in smoke tests) |
| `VOICE_MIN_UTTERANCE_S` | `1.0` | shorter audio → "can't tell" |
| `VOICE_PASSIVE_CAP` | `10` | max passive rows kept per person |

The borderline margin stays a code constant (0.05), shared with faces.

## Error handling

- speechbrain model fails to load → log once, speaker ID degrades to
  "can't tell" for every turn (fusion falls back to face); the robot keeps
  working as today.
- Voice store empty (nobody voice-enrolled yet) → `match_voice` returns
  `None` → "can't tell" → face-only behavior. No special casing needed.
- Passive backfill failures are logged and swallowed — never block a reply.

## Testing

- Fakes: `FakeSpeakerIdentifier` (scripted observations) and updated
  `FakeTranscriber` returning `Utterance`s in `tests/conftest.py`.
- `fusion.py`: table-driven test over the full truth table.
- Oracle scenarios: second enrolled person chimes in (their memories used,
  their notes saved); unknown voice asks to save a note (spoken refusal);
  voice contradicts face (voice wins); everything silent-degrades when the
  identifier returns "can't tell".
- Store: voices table round-trip, majority vote, passive cap eviction.
- Manual smoke test additions to `docs/testing.md`: two-person checklist
  mirroring the Phase 2 milestone.

## Privacy

Voice embeddings are biometrics like the face rows: local-only
(`data/lancedb/voices`), inspectable, deletable. Enrollment consent wording
already covers "I will remember you"; passive backfill only ever runs for
people who explicitly enrolled. Raw utterance audio is never persisted —
only the 192-dim embedding.

## Out of scope

Cross-person memory queries ("what did Alice say"), diarization within a
single utterance (two people talking over each other), voice-only greeting
(conversations still start with a face), wake word, fish-speech TTS.
