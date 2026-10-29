"""Public screens of faster nonlinear head training at matched noise scales."""
import hashlib
from pathlib import Path
import run_extended as e

r = e.r
r.SETTINGS.update({
    f'train_fast{batch}':r.Setting(f'train_fast{batch}',eta=1.,batch=batch)
    for batch in (256,384,512)
})
previous_hashes = r.source_hashes


def fast_hashes():
    hashes = previous_hashes()
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


r.source_hashes = fast_hashes

if __name__=='__main__':
    r.main()
