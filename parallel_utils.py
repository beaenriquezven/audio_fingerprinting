import os, pickle
from concurrent.futures import ProcessPoolExecutor
from inmemory_recognizer import InMemoryRecognizer

CACHE_DIR = '.hashcache'
os.makedirs(CACHE_DIR, exist_ok=True)
_recog = None

def init_worker(db_path):
    global _recog
    _recog = InMemoryRecognizer(db_path)

def extract_with_cache(path: str):
    cache = os.path.join(CACHE_DIR, os.path.basename(path) + '.pkl')
    if os.path.exists(cache):
        return path, pickle.load(open(cache,'rb'))
    tbl = _recog.extract_hashes(path)
    pickle.dump(tbl, open(cache,'wb'))
    return path, tbl

def recognize_one(arg):
    path, true, cat = arg
    _, tbl = extract_with_cache(path)
    pred, _ = _recog.recognize(path)
    return true, pred or 'no_match', cat

def warmup_cache(sample_list, db_path, n_workers=4):
    paths = [p for p,_,_ in sample_list]
    with ProcessPoolExecutor(n_workers, initializer=init_worker, initargs=(db_path,)) as exe:
        for _ in exe.map(extract_with_cache, paths): pass

def batch_recognize(sample_list, db_path, n_workers=4):
    with ProcessPoolExecutor(n_workers, initializer=init_worker, initargs=(db_path,)) as exe:
        for res in exe.map(recognize_one, sample_list): yield res

def batch_recognize_fast(sample_list, db_path, n_workers=4):
    warmup_cache(sample_list, db_path, n_workers)
    return batch_recognize(sample_list, db_path, n_workers)