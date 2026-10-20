from reachy_vec.config import Settings


def test_look_around_settle_default():
    assert Settings().look_around_settle_s == 0.8


def test_env_file_layering(tmp_path, monkeypatch):
    """Layered .env loading: repo .env overrides the profile .env; a real
    environment variable wins over both."""
    profile = tmp_path / "profile.env"
    repo = tmp_path / ".env"
    profile.write_text(
        "REACHY_VEC_ROBOT_HOST=profile-host\nREACHY_VEC_VOICE_THRESHOLD=0.41\n"
    )
    repo.write_text("REACHY_VEC_ROBOT_HOST=repo-host\n")
    monkeypatch.setenv("REACHY_VEC_WEATHER_LAT", "12.5")

    s = Settings(_env_file=(str(profile), str(repo)))
    assert s.robot_host == "repo-host"  # repo overrides profile
    assert s.voice_threshold == 0.41  # profile fills what repo lacks
    assert s.weather_lat == 12.5  # real env var wins over files
