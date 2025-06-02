import librosa
import numpy as np

def load_audio(filepath: str, target_sr: int = 44100) -> np.ndarray:
    """
    Carga un archivo WAV a mono y remuestrea a target_sr.
    Devuelve un array 1-D de float32.
    """
    y, _ = librosa.load(filepath, sr=target_sr, mono=True)
    return y