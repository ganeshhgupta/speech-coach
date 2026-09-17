"""prosody.py — acoustic measurements via Parselmouth (Praat bindings)."""
import numpy as np
import parselmouth
from .schema import ProsodyMetrics


def compute_prosody(wav_path: str) -> ProsodyMetrics:
    snd = parselmouth.Sound(wav_path)

    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    voiced = pitch_values[pitch_values > 0]

    if voiced.size > 0:
        pitch_mean = float(np.mean(voiced))
        pitch_stdev = float(np.std(voiced))
        pitch_cv = round(pitch_stdev / pitch_mean, 3) if pitch_mean else None
        pitch_mean = round(pitch_mean, 1)
        pitch_stdev = round(pitch_stdev, 1)
    else:
        pitch_mean = pitch_stdev = pitch_cv = None

    voiced_fraction = round(float(voiced.size / pitch_values.size), 3) if pitch_values.size else None

    intensity = snd.to_intensity()
    intensity_values = intensity.values[0]
    # Praat marks silent/undefined frames with a -300 dB sentinel, not NaN.
    intensity_values = intensity_values[np.isfinite(intensity_values) & (intensity_values > -200)]
    if intensity_values.size > 0:
        intensity_mean = round(float(np.mean(intensity_values)), 1)
        intensity_stdev = round(float(np.std(intensity_values)), 1)
    else:
        intensity_mean = intensity_stdev = None

    return ProsodyMetrics(
        pitch_mean_hz=pitch_mean,
        pitch_stdev_hz=pitch_stdev,
        pitch_cv=pitch_cv,
        intensity_mean_db=intensity_mean,
        intensity_stdev_db=intensity_stdev,
        voiced_fraction=voiced_fraction,
    )
