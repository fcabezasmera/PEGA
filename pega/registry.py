"""
pega.registry
=============
Auto-discovery registry for PEGA predictors.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pega.base import BasePredictor


class PredictorRegistry:
    """Singleton registry that discovers and exposes all PEGA predictors."""

    def __init__(self) -> None:
        self._predictors: dict[str, type[BasePredictor]] = {}
        self._discovered = False

    def _discover(self) -> None:
        if self._discovered:
            return
        from pega.base import BasePredictor
        import pega.predictors as predictors_pkg

        for module_info in pkgutil.iter_modules(predictors_pkg.__path__):
            module_name = f"pega.predictors.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"PEGA: could not import '{module_name}': {exc}", stacklevel=2)
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BasePredictor)
                    and obj is not BasePredictor
                    and obj.__module__ == module_name
                    and hasattr(obj, "name")
                ):
                    self._register(obj)
        self._discovered = True

    def _register(self, cls: type[BasePredictor]) -> None:
        if cls.name in self._predictors and self._predictors[cls.name] is not cls:
            warnings.warn(
                f"PEGA: predictor name '{cls.name}' is already registered. "
                f"Definition in {cls.__module__} will be ignored.",
                stacklevel=2,
            )
            return
        self._predictors[cls.name] = cls

    def list_all(self) -> list[type[BasePredictor]]:
        self._discover()
        return sorted(self._predictors.values(), key=lambda c: c.predictor_id)

    def list_available(self) -> list[type[BasePredictor]]:
        return [cls for cls in self.list_all() if cls.is_available()]

    def get(self, key: str | int) -> type[BasePredictor]:
        self._discover()
        if isinstance(key, str):
            if key not in self._predictors:
                raise KeyError(
                    f"No predictor named '{key}'. "
                    f"Available: {', '.join(sorted(self._predictors))}"
                )
            return self._predictors[key]
        if isinstance(key, int):
            for cls in self._predictors.values():
                if cls.predictor_id == key:
                    return cls
            ids = sorted(c.predictor_id for c in self._predictors.values())
            raise KeyError(f"No predictor with ID {key}. Valid IDs: {ids}")
        raise TypeError(f"Key must be str or int, got {type(key).__name__}.")

    def names(self) -> list[str]:
        self._discover()
        return sorted(self._predictors)

    def summary(self) -> str:
        """Return the formatted table shown by ``PEGA list``."""
        self._discover()

        col_w = 24  # display_name column width
        header = (
            f"  {'ID':>3}  {'Predictor':<{col_w}}  {'Status':<13}  "
            f"{'Type':<8}  Description"
        )
        sep = "  " + "─" * 82
        lines = [header, sep]

        for cls in self.list_all():
            status = "[available]" if cls.is_available() else "[unavailable]"
            lines.append(
                f"  {cls.predictor_id:>3}  {cls.display_name:<{col_w}}  "
                f"{status:<13}  {cls.category:<8}  {cls.description}"
            )

        n_ok  = len(self.list_available())
        n_all = len(self._predictors)
        lines += [sep, f"  {n_ok} of {n_all} predictors available."]
        return "\n".join(lines)

    def __repr__(self) -> str:
        self._discover()
        return (
            f"<PredictorRegistry "
            f"total={len(self._predictors)} "
            f"available={len(self.list_available())}>"
        )


registry = PredictorRegistry()
