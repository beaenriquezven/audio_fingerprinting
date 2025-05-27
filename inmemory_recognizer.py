import sqlite3
from collections import defaultdict, Counter
from math import log
from generator import Generator
from audio_loader import load_audio

class InMemoryRecognizer:
    def __init__(self, db_path: str, sr: int = 44100):
        self.sr = sr
        self.fg = Generator(sr=sr)
        # Construir índice y parámetros IDF/exclusión
        self.hash_index = self._load_index(db_path)
        self.idf = self._compute_idf()
        self.exclude_hashes = self._compute_excludes(threshold= 5)

    def _load_index(self, db_path: str) -> dict:
        """
        Carga el índice de la BD en memoria:
        retorna dict hash -> list of (track, time)
        """
        idx = defaultdict(list)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT hash, track, time FROM fingerprints")
        for h, track, t in cur:
            idx[h].append((track, t))
        con.close()
        return idx

    def _compute_idf(self) -> dict:
        """
        Calcula IDF por hash: log(N / df(h))
        """
        tracks = {tr for entries in self.hash_index.values() for tr, _ in entries}
        N = len(tracks) or 1
        return {h: log(N / len({tr for tr, _ in entries}))
                for h, entries in self.hash_index.items()}

    def _compute_excludes(self, threshold: int) -> set:
        """
        Marca hashes que aparecen en más de 'threshold' tracks.
        """
        counts = {h: len({tr for tr, _ in entries})
                  for h, entries in self.hash_index.items()}
        return {h for h, c in counts.items() if c > threshold}

    def extract_hashes(self, filepath: str) -> dict:
        y = load_audio(filepath, target_sr=self.sr)
        freqs, times, Sxx = self.fg.compute_spectrogram(y)
        peaks  = self.fg.get_peaks(Sxx, times, freqs)   # array (N,4)
      
        tbl = defaultdict(list)
        for h, t in self.fg.generate_hashes(peaks):
            tbl[h].append(t)
        return tbl

    def recognize(self, filepath: str) -> tuple:
        """
        Fallback simple: retorna la pista con más hashes en común (sin IDF ni offsets).
        """
        tbl = self.extract_hashes(filepath)
        counts = Counter()
        for h, times in tbl.items():
            if h in self.exclude_hashes:
                continue
            for track, _ in self.hash_index.get(h, []):
                counts[track] += len(times)
        if not counts:
            return None, 0
        # devolvemos el más frecuente
        return counts.most_common(1)[0]