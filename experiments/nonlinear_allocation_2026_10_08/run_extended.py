"""Additional public screens with larger classification heads and batches."""
import hashlib
from pathlib import Path
import run_search as r

r.SETTINGS.update({
    'train_full512':r.Setting('train_full512',features=32,hidden=12,batch=512),
    'train_medium384':r.Setting('train_medium384',features=16,hidden=8,batch=384,horizon=180),
    'train_small_long512':r.Setting('train_small_long512',batch=512,horizon=240),
})
base_hashes = r.source_hashes


def extended_hashes():
    hashes = base_hashes()
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


r.source_hashes = extended_hashes

if __name__=='__main__':
    r.main()
