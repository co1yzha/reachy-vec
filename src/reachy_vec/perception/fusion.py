"""Identity fusion: voice is the authority, face the tie-breaker; never guess.

Truth table (per utterance):
  voice known      -> that person (the speaker may be off-camera)
  voice unknown    -> anonymous (a stranger is talking, whoever is in frame)
  voice can't tell -> the recognized face, else anonymous
"""

from dataclasses import dataclass

from reachy_vec.perception.face import Observation


@dataclass(frozen=True)
class TurnIdentity:
    person_id: str | None
    name: str | None


ANONYMOUS = TurnIdentity(None, None)


def fuse(face_obs: Observation | None, voice_obs: Observation | None) -> TurnIdentity:
    if voice_obs is not None:
        if voice_obs.person_id is not None:
            return TurnIdentity(voice_obs.person_id, voice_obs.name)
        return ANONYMOUS
    if face_obs is not None and face_obs.person_id is not None:
        return TurnIdentity(face_obs.person_id, face_obs.name)
    return ANONYMOUS
