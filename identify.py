import os
import re
import sqlite3
import unicodedata
from audio_loader import load_audio
from generator import Generator


class Identify:
    """
    Reconocedor ultra-rápido que vuelca las huellas del sample en una tabla
    TEMPORAL y deja que SQLite ejecute en C un JOIN+GROUP BY para elegir
    la pista con más coincidencias.
    """
    def __init__(self, db_path: str, sr: int = 44100):
        self.conn = sqlite3.connect(db_path)
        self.cur  = self.conn.cursor()
        self.fg   = Generator(sr=sr)

    @staticmethod
    def _strip_accents(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @classmethod
    def normalize_key(cls, filename: str) -> str:
        # Extrae nombre base, quita sufijo numérico, puntos, underscores y acentos
        base = os.path.splitext(os.path.basename(filename))[0]
        base = re.sub(r'_(\d+)$', '', base)
        s = cls._strip_accents(base)
        s = s.replace('.', ' ').replace('_', ' ')
        s = re.sub(r'[^A-Za-z0-9 ]+', ' ', s)
        return ' '.join(s.split()).lower()

    def recognize(self, sample_path: str) -> (str, int):
        # 1) Extraer huellas
        y = load_audio(sample_path, target_sr=self.fg.sr)
        f_arr, t_arr, Sxx = self.fg.compute_spectrogram(y)
        tv, fv     = self.fg.get_peaks(Sxx, t_arr, f_arr)
        raw_hashes = list(self.fg.generate_hashes(tv, fv))  # [(hash, t), ...]

        if not raw_hashes:
            return None, 0

        # 2) Preparamos el CTE sample(hash)
        hashes = [h for h,_ in raw_hashes]
        # “VALUES (?), (?), …”
        values = ",".join(["(?)"] * len(hashes))
        cte = f"WITH sample(hash) AS (VALUES {values})"

        # 3) Query única muy rápida en C
        sql = f"""
        {cte}
        SELECT f.track, COUNT(*) AS cnt
        FROM fingerprints f
        JOIN sample s ON f.hash = s.hash
        GROUP BY f.track
        ORDER BY cnt DESC
        LIMIT 1;
        """
        self.cur.execute(sql, hashes)
        row = self.cur.fetchone()
        return (row[0], row[1]) if row else (None, 0)


            