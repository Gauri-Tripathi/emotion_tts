"""Provider registration and lookup."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from unitts.core.provider import BaseProvider

ProviderT = TypeVar("ProviderT", bound=type[BaseProvider])


class ProviderRegistry:
    """In-memory provider registry."""

    _providers: dict[str, type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[ProviderT], ProviderT]:
        """Register a provider class under ``name`` or its ``name`` attribute."""

        def decorator(provider_cls: ProviderT) -> ProviderT:
            provider_name = name or getattr(provider_cls, "name", provider_cls.__name__).lower()
            cls._providers[provider_name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseProvider]:
        """Return a provider class by name."""

        import unitts.providers  # noqa: F401

        key = name.lower()
        if key not in cls._providers:
            available = ", ".join(sorted(cls._providers))
            raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
        return cls._providers[key]

    @classmethod
    def all(cls) -> dict[str, type[BaseProvider]]:
        """Return all registered providers."""

        import unitts.providers  # noqa: F401

        return dict(cls._providers)
