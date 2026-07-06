"""Text-to-speech: synthesize on the Mac, play through the robot speaker (Phase 1).

Pluggable backends selected via settings.tts_backend:
- fish-speech: primary — best voice-clone quality (OpenAudio S1-mini)
- openvoice:   fallback if fish-speech latency on MPS is too high
- say:         macOS built-in, no cloning; dev/debug only
Voice cloning uses settings.voice_sample as the reference audio.
"""
