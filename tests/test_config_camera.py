from pathlib import Path

from reachy_vec.config import Settings


def test_vision_defaults():
    s = Settings()
    assert s.vision_model is None            # None => reuse llm_model
    assert s.vision_image_max_px == 1024
    assert s.photos_dir == Path("data") / "photos"


def test_vision_model_override_via_env(monkeypatch):
    monkeypatch.setenv("REACHY_VEC_VISION_MODEL", "gpt-4o")
    monkeypatch.setenv("REACHY_VEC_VISION_IMAGE_MAX_PX", "512")
    s = Settings()
    assert s.vision_model == "gpt-4o"
    assert s.vision_image_max_px == 512
