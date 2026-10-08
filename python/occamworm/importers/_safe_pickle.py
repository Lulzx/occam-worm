"""Read numeric fields from upstream pickles without importing or executing upstream code.

Unpickling can run arbitrary code, and the Leifer Lab classes are GPL-3.0, so we never import them. This
unpickler resolves only an explicit allowlist: numpy's array reconstruction and the two upstream classes,
which are replaced by an inert attribute container.
"""

from __future__ import annotations

import importlib
import pickle
from io import BytesIO
from typing import Any

import numpy as np

INERT_CLASSES = frozenset({("wormdatamodel.data.recording", "recording"), ("pumpprobe.Fconn", "Fconn")})


class Inert:
    """Stands in for an upstream class; holds its pickled attribute dict and nothing else."""

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("unexpected non-dict state for an upstream object")
        self.__dict__.update(state)


class AllowlistUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in INERT_CLASSES:
            return Inert
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            try:
                multiarray = importlib.import_module("numpy._core.multiarray")
            except ImportError:  # numpy < 2
                multiarray = importlib.import_module("numpy.core.multiarray")
            return getattr(multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_attrs(data: bytes) -> dict[str, Any]:
    obj = AllowlistUnpickler(BytesIO(data)).load()
    if not isinstance(obj, Inert):
        raise pickle.UnpicklingError(f"expected an upstream object, got {type(obj).__name__}")
    return dict(obj.__dict__)
