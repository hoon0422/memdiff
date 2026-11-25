"""Brute-force cosine index over embedding blobs, cached per (key, dim) until invalidated."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence

import numpy as np

# (id, step, embedding blob) in storage order; the order decides tie-breaking.
IndexRow = tuple[str, int, bytes | None]


def encode_embedding(vec: Sequence[float] | np.ndarray | None) -> bytes | None:
    """float32, L2-normalized (when non-zero) bytes; `None` stays NULL."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm if norm > 0 else arr).tobytes()


class NumpyVectorIndex:
    def __init__(self, load: Callable[[Hashable], Iterable[IndexRow]]) -> None:
        self._load = load
        self._cache: dict[tuple[Hashable, int], tuple[list[str], np.ndarray, np.ndarray]] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    def _entry(self, key: Hashable, dim: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        if (key, dim) not in self._cache:
            ids: list[str] = []
            steps: list[int] = []
            vecs: list[np.ndarray] = []
            for rid, step, blob in self._load(key):
                vec = np.frombuffer(blob, dtype=np.float32) if blob else None
                if vec is None or vec.shape != (dim,):
                    continue
                ids.append(rid)
                steps.append(step)
                vecs.append(vec)
            matrix = np.vstack(vecs) if vecs else np.empty((0, dim), dtype=np.float32)
            self._cache[(key, dim)] = (ids, np.asarray(steps, dtype=np.int64), matrix)
        return self._cache[(key, dim)]

    def top(
        self,
        query: Sequence[float] | np.ndarray,
        *,
        key: Hashable = None,
        top_k: int,
        recency_decay: float = 0.0,
        min_score: float = -1.0,
    ) -> list[tuple[str, float]]:
        """Best `top_k` (id, cosine score) pairs, highest first."""
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.size == 0 or top_k <= 0:
            return []
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        ids, steps, matrix = self._entry(key, int(q.shape[0]))
        if not ids:
            return []
        scores = matrix @ q
        if recency_decay > 0.0:
            scores = scores * np.exp(-recency_decay * np.maximum(0, int(steps.max()) - steps))
        valid = np.flatnonzero(scores >= min_score)
        order = np.argsort(scores[valid])[::-1][:top_k]
        return [(ids[int(valid[int(p)])], float(scores[valid[int(p)]])) for p in order]
