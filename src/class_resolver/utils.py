"""Utilities for the resolver."""

from __future__ import annotations

import collections.abc
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, overload

if TYPE_CHECKING:
    import click  # pragma: no cover

__all__ = [
    "Hint",
    "HintOrType",
    "HintType",
    "InstOrType",
    "Lookup",
    "LookupOrType",
    "LookupType",
    "OneOrManyHintOrType",
    "OneOrManyOptionalKwargs",
    "OptionalKwargs",
    "X",
    "Y",
    "get_subclasses",
    "make_callback",
    "normalize_string",
    "normalize_with_default",
    "same_module",
    "upgrade_to_sequence",
]

logger = logging.getLogger(__name__)

X = TypeVar("X")
Y = TypeVar("Y")

#: A type annotation for either an instance of X or a class of X
InstOrType: TypeAlias = X | type[X]
#: A type annotation for either an instance of X or name a class X
Lookup: TypeAlias = str | X

LookupType: TypeAlias = Lookup[type[X]]
LookupOrType: TypeAlias = Lookup[InstOrType[X]]
Hint: TypeAlias = Lookup[X] | None
HintType: TypeAlias = Hint[type[X]]
HintOrType: TypeAlias = Hint[InstOrType[X]]
OptionalKwargs: TypeAlias = Mapping[str, Any] | None
OneOrSequence: TypeAlias = X | Sequence[X]
OneOrManyHintOrType: TypeAlias = OneOrSequence[HintOrType[X]] | None
OneOrManyOptionalKwargs: TypeAlias = OneOrSequence[OptionalKwargs] | None


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
        return (x,)
    elif isinstance(x, collections.abc.Sequence):
        return x
    else:
        return (x,)


def make_callback(f: Callable[[X], Y]) -> Callable[[click.Context, click.Parameter, X], Y]:
    """Make a click-appropriate callback."""

    def _callback(_ctx: click.Context, _param: click.Parameter, value: X) -> Y:
        return f(value)

    return _callback


# docstr-coverage:excused `overload`
@overload
def normalize_with_default(
    choice: None,
    kwargs: OptionalKwargs = ...,
    default: None = ...,
    default_kwargs: OptionalKwargs = ...,
) -> tuple[None, OptionalKwargs]: ...


# docstr-coverage:excused `overload`
@overload
def normalize_with_default(
    choice: None,
    kwargs: OptionalKwargs = ...,
    default: Y = ...,
    default_kwargs: OptionalKwargs = ...,
) -> tuple[Y, OptionalKwargs]: ...


# docstr-coverage:excused `overload`
@overload
def normalize_with_default(
    choice: X,
    kwargs: OptionalKwargs = ...,
    default: Y | None = ...,
    default_kwargs: OptionalKwargs = ...,
) -> tuple[X, OptionalKwargs]: ...


def normalize_with_default(
    choice: X | None,
    kwargs: OptionalKwargs = None,
    default: Y | None = None,
    default_kwargs: OptionalKwargs = None,
) -> tuple[X | Y | None, OptionalKwargs]:
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
    if choice is not None:
        return choice, kwargs or default_kwargs
    if default is None:
        raise ValueError("If choice is None, a default has to be provided.")
    if kwargs is not None:
        logger.warning(
            f"No choice was provided, but kwargs={kwargs} is not None. Will use the default choice={default} "
            f"with its default_kwargs={default_kwargs}. If you want the explicitly provided kwargs to be used,"
            f" explicitly provide choice={default} instead of None."
        )
    return default, default_kwargs
