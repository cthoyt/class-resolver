# -*- coding: utf-8 -*-

"""Lookup and instantiate classes with style."""

from .api import (
    Hint,
    HintOrType,
    HintType,
    InstOrType,
    KeywordArgumentError,
    Lookup,
    LookupOrType,
    LookupType,
    OptionalKwargs,
    Resolver,
    UnexpectedKeywordError,
    get_cls,
    get_subclasses,
    normalize_string,
)

__all__ = [
    # Type Hints
    "InstOrType",
    "Lookup",
    "LookupType",
    "LookupOrType",
    "Hint",
    "HintType",
    "HintOrType",
    # Classes
    "Resolver",
    # Utilities
    "get_cls",
    "get_subclasses",
    "normalize_string",
    # Exceptions
    "KeywordArgumentError",
    "UnexpectedKeywordError",
]
