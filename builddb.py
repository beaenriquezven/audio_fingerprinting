import sqlite3
import unicodedata, re

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
