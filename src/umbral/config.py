"""Configuration: a base YAML file plus dotted CLI overrides, frozen after loading.

Example::

    cfg = load_config("configs/base.yaml", ["seed=3", "run.name=prueba"])
    cfg.seed            # 3
    cfg.run.name        # "prueba"
    cfg["run"]["name"]  # same thing

Overrides may only target keys that already exist in the base file, so a typo
fails loudly instead of silently creating a new key.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(Mapping[str, Any]):
    """Immutable nested mapping with attribute access. Lists become tuples."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", {str(k): _freeze(v) for k, v in data.items()})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"config has no key {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("Config is immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("Config is immutable")

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, mutable deep copy (dicts and lists)."""
        return {k: _thaw(v) for k, v in self._data.items()}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value if isinstance(value, Config) else Config(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value``; the value is typed with YAML rules (3 -> int, true -> bool)."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise ValueError(f"override {text!r} has an invalid key")
    return key.split("."), yaml.safe_load(raw) if raw.strip() else ""


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted overrides in place to a plain dict and return it."""
    for text in overrides:
        path, value = parse_override(text)
        node = data
        for i, part in enumerate(path):
            if not isinstance(node, dict) or part not in node:
                where = ".".join(path[: i + 1])
                raise KeyError(f"override {text!r}: unknown config key {where!r}")
            if i == len(path) - 1:
                if isinstance(node[part], dict):
                    raise KeyError(f"override {text!r}: {'.'.join(path)!r} is a section, not a value")
                node[part] = value
            else:
                node = node[part]
    return data


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return Config(apply_overrides(data, overrides))
