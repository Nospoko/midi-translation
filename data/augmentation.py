import random


def change_speed(record: dict, factor: float = None) -> dict:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + random.random() * change_range

    record["start"] = [value / factor for value in record["start"]]
    record["end"] = [value / factor for value in record["end"]]
    record["duration"] = [x - y for x, y in zip(record["end"], record["start"])]
    return record


def pitch_shift(pitch: list[int], shift_threshold: int = 5) -> list[int]:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, min(pitch) - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - max(pitch))

    if low_shift > high_shift:
        shift = 0
    else:
        shift = random.randint(low_shift, high_shift + 1)
    pitch = [value + shift for value in pitch]

    return pitch
