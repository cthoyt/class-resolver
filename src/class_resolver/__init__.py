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
from .func import FunctionResolver

__all__ = [
    # Type Hints
    "InstOrType",
    "Lookup",
    "LookupType",
    "LookupOrType",
    "Hint",
    "HintType",
    "HintOrType",
    "OptionalKwargs",
    # Classes
    "Resolver",
    "FunctionResolver",
    # Utilities
    "get_cls",
    "get_subclasses",
    "normalize_string",
    # Exceptions
    "KeywordArgumentError",
    "UnexpectedKeywordError",
]
