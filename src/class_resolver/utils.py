# -*- coding: utf-8 -*-

"""Utilities for the resolver."""

import collections.abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    import click  # pragma: no cover

__all__ = [
    # Type Hints
    "X",
    "Hint",
    "HintOrType",
    "HintType",
    "InstOrType",
    "Lookup",
    "LookupOrType",
    "LookupType",
    "OptionalKwargs",
    "OneOrManyHintOrType",
    "OneOrManyOptionalKwargs",
    # Functions
    "get_subclasses",
    "normalize_string",
    "upgrade_to_sequence",
    "make_callback",
    "same_module",
]

X = TypeVar("X")
Y = TypeVar("Y")

#: A type annotation for either an instance of X or a class of X
InstOrType = Union[X, Type[X]]
#: A type annotation for either an instance of X or name a class X
Lookup = Union[str, X]

LookupType = Lookup[Type[X]]
LookupOrType = Lookup[InstOrType[X]]
Hint = Optional[Lookup[X]]
HintType = Hint[Type[X]]
HintOrType = Hint[InstOrType[X]]
OptionalKwargs = Optional[Mapping[str, Any]]
OneOrSequence = Union[X, Sequence[X]]
OneOrManyHintOrType = Optional[OneOrSequence[HintOrType[X]]]
OneOrManyOptionalKwargs = Optional[OneOrSequence[OptionalKwargs]]


def get_subclasses(
    cls: Type[X],
    exclude_private: bool = True,
    exclude_external: bool = True,
) -> Iterable[Type[X]]:
    """Get all subclasses.

    :param cls: The ancestor class
    :param exclude_private: If true, will skip any class that comes from a module
        starting with an underscore (i.e., a private module). This is typically
        done when having shadow duplicate classes implemented in C
    :param exclude_external: If true, will exclude any class that does not originate
        from the same package as the base class.
    :yields: Descendant classes of the ancestor class
    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        if exclude_private:
            if any(part.startswith("_") for part in subclass.__module__.split(".")):
                continue
            if subclass.__name__.startswith("_"):
                continue
        if exclude_external and not same_module(cls, subclass):
            continue
        yield subclass


def same_module(cls1: type, cls2: type) -> bool:
    """Return if two classes come from the same module via the ``__module__`` attribute."""
    return cls1.__module__.split(".")[0] == cls2.__module__.split(".")[0]


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    if suffix is not None and s.endswith(suffix.lower()):
        return s[: -len(suffix)]
    return s.strip()


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence.

    :param x: A literal or sequence of literals (don't consider a string x as a sequence)
    :return: If a literal was given, a one element tuple with it in it. Otherwise, return the given value.

    >>> upgrade_to_sequence(1)
    (1,)
    >>> upgrade_to_sequence((1, 2, 3))
    (1, 2, 3)
    >>> upgrade_to_sequence("test")
    ('test',)
    >>> upgrade_to_sequence(tuple("test"))
    ('t', 'e', 's', 't')
    """
    if isinstance(x, str):
        return (x,)  # type: ignore
    elif isinstance(x, collections.abc.Sequence):
        return x
    else:
        return (x,)


def make_callback(f: Callable[[X], Y]) -> Callable[["click.Context", "click.Parameter", X], Y]:
    """Make a click-appropriate callback."""

    def _callback(_ctx: "click.Context", _param: "click.Parameter", value: X) -> Y:
        return f(value)

    return _callback
