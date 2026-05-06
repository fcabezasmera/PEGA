"""
pega.registry
=============
Auto-discovery registry for PEGA predictors.

The registry scans ``pega/predictors/`` at first access and registers every
class that inherits from :class:`~pega.base.BasePredictor`.  Adding a new
predictor module to that directory is sufficient — no manual registration
is needed anywhere else.

Usage
-----
    from pega.registry import registry

    registry.list_all()        # all predictors, sorted by predictor_id
    registry.list_available()  # only those whose dependencies are met
    registry.get("ampnet")     # by name
    registry.get(1)            # by predictor_id
    print(registry.summary())  # formatted table — same as `pega list`
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
    """Singleton registry that discovers and exposes all PEGA predictors.

    Use the module-level :data:`registry` instance rather than
    instantiating this class directly.
    """

    def __init__(self) -> None:
        self._predictors: dict[str, type[BasePredictor]] = {}
        self._discovered = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Import all modules under ``pega.predictors`` and collect classes.

        Called lazily so that ``import pega`` remains fast.
        """
        if self._discovered:
            return

        from pega.base import BasePredictor
        import pega.predictors as predictors_pkg

        for module_info in pkgutil.iter_modules(predictors_pkg.__path__):
            module_name = f"pega.predictors.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"PEGA: could not import '{module_name}': {exc}",
                    stacklevel=2,
                )
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_all(self) -> list[type[BasePredictor]]:
        """Return all registered predictors sorted by ``predictor_id``."""
        self._discover()
        return sorted(self._predictors.values(), key=lambda c: c.predictor_id)

    def list_available(self) -> list[type[BasePredictor]]:
        """Return predictors whose runtime dependencies are satisfied."""
        return [cls for cls in self.list_all() if cls.is_available()]

    def get(self, key: str | int) -> type[BasePredictor]:
        """Retrieve a predictor class by name or integer ID.

        Raises
        ------
        KeyError
            If no predictor matches ``key``.
        TypeError
            If ``key`` is not a ``str`` or ``int``.
        """
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

        raise TypeError(
            f"Key must be str (name) or int (predictor_id), "
            f"got {type(key).__name__}."
        )

    def names(self) -> list[str]:
        """Return a sorted list of all registered predictor names."""
        self._discover()
        return sorted(self._predictors)

    def summary(self) -> str:
        """Return the formatted table shown by ``pega list``."""
        self._discover()

        header = (
            f"{'ID':>3}  {'Name':<22}  {'Status':<13}  {'Category':<8}  Description"
        )
        sep = "-" * 80
        lines = [header, sep]

        for cls in self.list_all():
            status = "[available]" if cls.is_available() else "[unavailable]"
            lines.append(
                f"{cls.predictor_id:>3}  {cls.name:<22}  {status:<13}  "
                f"{cls.category:<8}  {cls.description}"
            )

        n_ok = len(self.list_available())
        n_all = len(self._predictors)
        lines += [sep, f"{n_ok} of {n_all} predictors available."]
        return "\n".join(lines)

    def __repr__(self) -> str:
        self._discover()
        return (
            f"<PredictorRegistry "
            f"total={len(self._predictors)} "
            f"available={len(self.list_available())}>"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = PredictorRegistry()
