"""audio_io.py — decode any uploaded audio container to a mono 16kHz WAV."""
import av
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def decode_to_wav(src_path: str, dst_path: str) -> float:
    """Decode src_path (any container av can read) to mono 16kHz PCM WAV at dst_path.
    Returns duration in seconds.
    """
    container = av.open(src_path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)

    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))
    container.close()

    if not chunks:
        raise ValueError("No audio data decoded from file")

    samples = np.concatenate(chunks).astype(np.int16)
    sf.write(dst_path, samples, TARGET_SR, subtype="PCM_16")
    return len(samples) / TARGET_SR
