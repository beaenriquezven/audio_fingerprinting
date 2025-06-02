import sqlite3
import unicodedata, re
from audio_loader import load_audio
from generator import Generator
import os
import argparse

class BuildDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.executescript("""
        DROP TABLE IF EXISTS fingerprints;
        CREATE TABLE fingerprints(
          hash   TEXT,
          track  TEXT,
          time   REAL
        );
        CREATE INDEX idx_hash ON fingerprints(hash);
        """)
        con.commit()
        con.close()

    def add_batch(self, rows):
        """Inserta una lista de tuplas (hash, track, time)."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.executemany(
            "INSERT INTO fingerprints(hash,track,time) VALUES (?,?,?)",
            rows
        )
        con.commit()
        con.close()

    def normalize_key(self, s: str) -> str:
        s = unicodedata.normalize('NFKD', s.lower())
        return re.sub(r'\W+', '_', s)
    
def main():
        parser = argparse.ArgumentParser(
            description="Build the fingerprint database from an audio folder."
        )
        parser.add_argument(
            "-i", "--input",
            required=True,
            help="Path to the folder containing the .wav files")
        parser.add_argument(
            "-o", "--output",
            required=True,
            help="Path to the SQLite file where the fingerprints will be saved.")
        args = parser.parse_args()

        music_folder = args.input
        db_path = args.output

        # If there was a previous DB, we delete it to recreate from scratch
        if os.path.exists(db_path):
            os.remove(db_path)

        #Instantiate the builder and the generator 
        builder = BuildDB(db_path)
        fg = Generator(sr=44100)

        # Recursively traverse all .wav files in music_folder
        for root, _, files in os.walk(music_folder):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue

                raw_key = os.path.splitext(fname)[0]
                track_key = builder.normalize_key(raw_key)

                wav_path = os.path.join(root, fname)
                # Carga de audio (target_sr=44100)
                y = load_audio(wav_path, target_sr=44100)

                # Cálculo de espectrograma, detección de picos, generación de hashes
                freqs, times, Sxx = fg.compute_spectrogram(y)
                peaks = fg.get_peaks(Sxx, times, freqs)
                rows = [(h, track_key, t) for h, t in fg.generate_hashes(peaks)]

                # Insertamos en lotes en la base de datos
                builder.add_batch(rows)

        print(f"[builddb] Database created at '{db_path}', with all fingerprints inserted.")

if __name__ == "__main__":
    main()
