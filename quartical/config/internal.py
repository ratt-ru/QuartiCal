import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
from quartical.config.preprocess import _sky_model


@dataclass
class Internals:
    model_columns: Set[str] = field(default_factory=lambda: set())
    sky_models: Set[_sky_model] = field(default_factory=lambda: set())
    recipe: Dict[int, List[Any]] = field(default_factory=lambda: dict())
    predict: bool = False
    ms_ncorr: int = 4
    feed_type: str = "linear"
    phase_dir: Any = np.zeros(2, dtype=np.float64)
