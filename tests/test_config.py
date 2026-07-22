from reachy_vec.config import Settings


def test_look_around_settle_default():
    assert Settings().look_around_settle_s == 0.8
