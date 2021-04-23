# -*- coding: utf-8 -*-

"""Lookup and instantiate classes with style."""

from .api import Hint, HintOrType, HintType, Resolver, get_cls, get_subclasses, normalize_string

__all__ = [
    'Hint',
    'HintType',
    'HintOrType',
    'Resolver',
    'get_cls',
    'get_subclasses',
    'normalize_string',
]
