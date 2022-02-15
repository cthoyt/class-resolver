import inspect
from typing import Any, Iterable, Mapping, Optional, Tuple, Type, TypeVar

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
            resolver.base: resolver for resolver in resolvers
        }
        self.names = {resolver.suffix: resolver for cls, resolver in self.resolvers.items()}

    def check_kwargs(self, base: Type[X], query: Hint[X], kwargs: OptionalKwargs) -> bool:
        main = self.resolvers[base]
        signature = main.signature(query)
        parameters = signature.parameters
        for key, parameter, related_key, related_parameter in _iter_params(parameters):
            annotation = parameter.annotation
            next_resolver = self.names.get(key)
            if next_resolver is None:
                if key not in kwargs:
                    if parameter.default is parameter.empty:
                        raise ValueError(f"{key} without default not given")
                else:
                    if not isinstance(kwargs[key], parameter.annotation):
                        raise TypeError
            elif not is_hint(annotation, next_resolver.base):
                raise TypeError(
                    f"{key} has bad annotation {annotation} wrt resolver {next_resolver}"
                )
            elif not self.check_kwargs(
                next_resolver.base,
                next_resolver.lookup(kwargs[key]),
                kwargs[related_key],
            ):
                return False
            else:
                continue


def _iter_params(
    parameters,
) -> Iterable[Tuple[str, inspect.Parameter, str, Optional[inspect.Parameter]]]:
    kwarg_map = {}
    for key in parameters.items():
        related_key = f"{key}_kwargs"
        if related_key in parameters:
            kwarg_map[key] = related_key
    for key in parameters:
        related_key = f"{key}_kwargs"
        if key in kwarg_map:
            continue
        yield key, parameters[key], related_key, parameters.get(related_key)
