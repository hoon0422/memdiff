from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class EmbeddingResult(BaseModel):
    """L2-normalized vectors, one per input text, in input order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectors: list[np.ndarray]
    total_tokens: int = 0
