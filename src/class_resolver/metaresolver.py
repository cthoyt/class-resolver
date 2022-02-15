from typing import Any, Iterable, Mapping, Type, TypeVar

from .api import ClassResolver
from .utils import Hint, OptionalKwargs

X = TypeVar("X")

__all__ = [
    "Metaresolver",
]


def is_hint(hint: Any, cls: Type[X]) -> bool:
    return hint == Hint[cls]


class Metaresolver:
    """A resolver of resolvers."""

    def __init__(self, resolvers: Iterable[ClassResolver]):
        self.resolvers: Mapping[Type, ClassResolver] = {
            resolver.base: resolver
            for resolver in resolvers
        }
        self.names = {
            resolver.normalize_cls(cls): resolver
            for cls, resolver in self.resolvers.items()
        }

    def check_kwargs(self, base: Type[X], query: Hint[X], kwargs: OptionalKwargs) -> bool:
        main = self.resolvers[base]
        signature = main.signature(query)
        parameters = signature.parameters

        for name, parameter in parameters.items():
            annotation = parameter.annotation

            next_resolver = self.names.get(name)
            if next_resolver is None:
                raise NotImplementedError
            elif not is_hint(annotation, next_resolver.base):
                raise TypeError
            else:
                raise NotImplementedError
