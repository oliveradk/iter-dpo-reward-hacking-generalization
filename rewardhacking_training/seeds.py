# One master run seed -> independent per-purpose seeds (each env's prompt order, the
# interleaved batch schedule), so changing one stream never shifts another. Keys are
# hashed with blake2b, not `hash()` (randomized per process).
from __future__ import annotations

import hashlib

SEED_MODULUS = 2**31
"""seeds are in [0, 2^31): small enough for every backend's seed field"""

STREAMS = ("data/impossible_mbpp", "data/nl_gameable", "data/schedule")


def derive_seed(*parts: object) -> int:
    """A seed in `[0, SEED_MODULUS)` that is a stable function of `parts` (joined with `/`)."""
    key = "/".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % SEED_MODULUS


def run_seeds(seed: int) -> dict[str, int]:
    """The per-stream seeds of master run seed `seed` (`STREAMS` -> seed)."""
    return {stream: derive_seed(seed, stream) for stream in STREAMS}
