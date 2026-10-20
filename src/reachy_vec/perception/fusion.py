"""Per-turn identity from the speaker's voice; never guess.

  voice known      -> that person
  voice unknown    -> anonymous (a stranger is talking)
  voice can't tell -> anonymous
"""

from dataclasses import dataclass

from reachy_vec.perception.voice import Observation


@dataclass(frozen=True)
class TurnIdentity:
    person_id: str | None
    name: str | None


ANONYMOUS = TurnIdentity(None, None)


def from_voice(voice_obs: Observation | None) -> TurnIdentity:
    if voice_obs is not None and voice_obs.person_id is not None:
        return TurnIdentity(voice_obs.person_id, voice_obs.name)
    return ANONYMOUS
