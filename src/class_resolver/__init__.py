# -*- coding: utf-8 -*-

"""Lookup and instantiate classes with style."""

from .api import KeywordArgumentError, Resolver, UnexpectedKeywordError, get_cls
from .func import FunctionResolver
from .utils import (
    Hint,
    HintOrType,
    HintType,
    InstOrType,
    Lookup,
    LookupOrType,
    LookupType,
    OptionalKwargs,
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
