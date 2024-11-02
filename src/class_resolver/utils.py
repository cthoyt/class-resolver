"""Utilities for the resolver."""

from __future__ import annotations

import collections.abc
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    import click  # pragma: no cover

__all__ = [
    # Type Hints
    "X",
    "Y",
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
    "normalize_with_default",
]

logger = logging.getLogger(__name__)

X = TypeVar("X")
Y = TypeVar("Y")

#: A type annotation for either an instance of X or a class of X
InstOrType = Union[X, type[X]]
#: A type annotation for either an instance of X or name a class X
Lookup = Union[str, X]

LookupType = Lookup[type[X]]
LookupOrType = Lookup[InstOrType[X]]
Hint = Optional[Lookup[X]]
HintType = Hint[type[X]]
HintOrType = Hint[InstOrType[X]]
OptionalKwargs = Optional[Mapping[str, Any]]
OneOrSequence = Union[X, Sequence[X]]
OneOrManyHintOrType = Optional[OneOrSequence[HintOrType[X]]]
OneOrManyOptionalKwargs = Optional[OneOrSequence[OptionalKwargs]]


def is_private(class_name: str, module_name: str, main_is_private: bool = True) -> bool:
    """
    Decide whether a class in a module is considered private.

    :param class_name:
        the class name, i.e., `cls.__name__`
    :param module_name:
        the module name, i.e., `cls.__module__`
    :param main_is_private:
        whether the `__main__` module is considered private

    :return:
        whether the class should be considered private
    """
    # note: this method has been separated for better testability
    if class_name.startswith("_"):
        return True
    if not main_is_private and module_name.startswith("__main__"):
        return False
    if any(part.startswith("_") for part in module_name.split(".")):
        return True
    return False


def get_subclasses(
    cls: type[X],
    exclude_private: bool = True,
    exclude_external: bool = True,
    main_is_private: bool = True,
) -> Iterable[type[X]]:
    """Get all subclasses.

    :param cls: The ancestor class
    :param exclude_private: If true, will skip any class that comes from a module
        starting with an underscore (i.e., a private module). This is typically
        done when having shadow duplicate classes implemented in C
    :param exclude_external: If true, will exclude any class that does not originate
        from the same package as the base class.
    :param main_is_private: If true, __main__ is considered a private module.
    :yields: Descendant classes of the ancestor class
    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        if exclude_private and is_private(
            class_name=subclass.__name__,
            module_name=subclass.__module__,
            main_is_private=main_is_private,
        ):
            continue
        if exclude_external and not same_module(cls, subclass):
            continue
        yield subclass


def same_module(cls1: type, cls2: type) -> bool:
    """Return if two classes come from the same module via the ``__module__`` attribute."""
    return cls1.__module__.split(".")[0] == cls2.__module__.split(".")[0]


def normalize_string(s: str, *, suffix: str | None = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    if suffix is not None and s.endswith(suffix.lower()):
        return s[: -len(suffix)]
    return s.strip()


def upgrade_to_sequence(x: X | Sequence[X]) -> Sequence[X]:
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


def make_callback(f: Callable[[X], Y]) -> Callable[[click.Context, click.Parameter, X], Y]:
    """Make a click-appropriate callback."""

    def _callback(_ctx: click.Context, _param: click.Parameter, value: X) -> Y:
        return f(value)

    return _callback


def normalize_with_default(
    choice: HintOrType[X],
    kwargs: OptionalKwargs = None,
    default: HintOrType[X] = None,
    default_kwargs: OptionalKwargs = None,
) -> tuple[HintOrType[X], OptionalKwargs]:
    """
    Normalize a choice for class resolver, with default options.

    :param choice:
        the choice. If None, use the default instead.
    :param kwargs:
        the keyword-based parameters for instantiation. Will only be used if choice is *not* None.
    :param default:
        the default choice. Used of choice=None.
    :param default_kwargs:
        the default keyword-based parameters

    :raises ValueError:
        if choice and default both are None

    :return:
        a pair (hint, optional kwargs).
    """
    if choice is None:
        if default is None:
            raise ValueError("If choice is None, a default has to be provided.")
        choice = default
        if kwargs is not None:
            logger.warning(
                f"No choice was provided, but kwargs={kwargs} is not None. Will use the default choice={default} "
                f"with its default_kwargs={default_kwargs}. If you want the explicitly provided kwargs to be used,"
                f" explicitly provide choice={default} instead of None."
            )
        kwargs = default_kwargs
    return choice, kwargs
