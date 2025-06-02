import os
import re
import sqlite3
import unicodedata
from audio_loader import load_audio
from generator import Generator
import argparse
from inmemory_recognizer import InMemoryRecognizer


class Identify:
    def __init__(self, db_path: str, sr: int = 44100):
        self.conn = sqlite3.connect(db_path)
        self.cur  = self.conn.cursor()
        self.fg   = Generator(sr=sr)


def main():
    parser = argparse.ArgumentParser(
        description="Recognize which track in the database best matches the given sample."
    )
    parser.add_argument(
        "-d", "--database",
        required=True,
        help="Path to the SQLite database file containing fingerprints (e.g., database_file.sqlite)."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the .wav file you want to identify."
    )
    args = parser.parse_args()

    db_path = args.database
    sample_path = args.input

    # Aquí usamos InMemoryRecognizer (igual que en el notebook)
    rec = InMemoryRecognizer(db_path, sr=44100)
    track, count = rec.recognize(sample_path)

    if track is None:
        print("[identify] No matches found.")
    else:
        print(f"[identify] Most likely track: '{track}' with {count} matches.")

if __name__ == "__main__":
    main()
         