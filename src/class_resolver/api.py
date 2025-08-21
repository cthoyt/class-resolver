"""Resolve classes."""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Generic, TypeVar

from .base import BaseResolver
from .utils import (
    HintOrType,
    HintType,
    OneOrManyHintOrType,
    OneOrManyOptionalKwargs,
    get_subclasses,
    normalize_string,
    upgrade_to_sequence,
)

__all__ = [
    "ClassResolver",
    "KeywordArgumentError",
    "Resolver",
    "UnexpectedKeywordError",
    "get_cls",
]

X = TypeVar("X")

logger = logging.getLogger(__name__)


class KeywordArgumentError(TypeError):
    """Thrown when missing a keyword-only argument."""

    def __init__(self, cls: type, s: str) -> None:
        """Initialize the error.

        :param cls: The class that was trying to be instantiated
        :param s: The string describing the original type error
        """
        self.cls = cls
        self.name = s.rstrip("'").rsplit("'", 1)[1]

    def __str__(self) -> str:
        return f"{self.cls.__name__}: __init__() missing 1 required keyword-only argument: '{self.name}'"


class UnexpectedKeywordError(TypeError):
    """Thrown when no arguments were expected."""

    def __init__(self, cls: type) -> None:
        """Initialize the error.

        :param cls: The class that was trying to be instantiated
        """
        self.cls = cls

    def __str__(self) -> str:
        return f"{self.cls.__name__} did not expect any keyword arguments"


MISSING_ARGS = [
    "takes no parameters",  # in 3.6
    "takes no arguments",  # > 3.7
]


class ClassResolver(Generic[X], BaseResolver[type[X], X]):
    """Resolve from a list of classes."""

    #: The base class
    base: type[X]
    #: The shared suffix fo all classes derived from the base class
    suffix: str
    #: The variable name to look up synonyms in classes that are registered with this resolver
    synonyms_attributes: list[str]

    def __init__(
        self,
        classes: Collection[type[X]] | None = None,
        *,
        base: type[X],
        default: type[X] | None = None,
        suffix: str | None = None,
        synonyms: Mapping[str, type[X]] | None = None,
        synonym_attribute: str | list[str] | None = "synonyms",
        base_as_suffix: bool = True,
        location: str | None = None,
    ) -> None:
        """Initialize the resolver.

        :param classes: A list of classes
        :param base: The base class
        :param default: The default class
        :param suffix: The optional shared suffix of all instances. If not none, will
            override ``base_as_suffix``.
        :param synonyms: The optional synonym dictionary
        :param synonym_attribute:
            The attribute or list of attributes to look in each class for synonyms.
            Defaults to ``synonyms``. Explicitly set to None to turn off synonym lookup.
        :param base_as_suffix: Should the base class's name be used as the suffix if
            none is given? Defaults to true.
        :param location: The location used to document the resolver in sphinx
        """
        self.base = base
        if isinstance(synonym_attribute, str):
            self.synonyms_attributes = [synonym_attribute]
        elif isinstance(synonym_attribute, list):
            self.synonyms_attributes = synonym_attribute
        elif synonym_attribute is None:
            self.synonyms_attributes = []
        else:
            raise TypeError

        if suffix is not None:
            if suffix == "":
                suffix = None
        elif base_as_suffix:
            suffix = normalize_string(self.base.__name__)
        super().__init__(
            elements=classes,
            synonyms=synonyms,
            default=default,
            suffix=suffix,
            location=location,
        )

    def extract_name(self, element: type[X]) -> str:
        """Get the name for an element."""
        return element.__name__

    @property
    def synonym_attribute(self) -> str | None:
        """Get the synonnym attribute for the class used for synonym lookup."""
        warnings.warn(
            "synonym_attribute is deprecated. Access the synonym_attributes list directly instead",
            DeprecationWarning,
            stacklevel=2,
        )
        lll = len(self.synonyms_attributes)
        if lll == 0:
            return None
        elif lll == 1:
            return self.synonyms_attributes[0]
        else:
            raise ValueError

    def extract_synonyms(self, element: type[X]) -> Collection[str]:
        """Get synonyms from an element."""
        rv = []
        for attribute in self.synonyms_attributes:
            x = getattr(element, attribute, None)
            if x is None:
                pass
            elif isinstance(x, str):
                rv.append(x)
            else:
                rv.extend(x)  # it's a list
        return rv

    @classmethod
    def from_subclasses(
        cls,
        base: type[X],
        *,
        skip: Collection[type[X]] | None = None,
        exclude_private: bool = True,
        exclude_external: bool = True,
        **kwargs: Any,
    ) -> ClassResolver[X]:
        """Make a resolver from the subclasses of a given class.

        :param base: The base class whose subclasses will be indexed
        :param skip: Any subclasses to skip (usually good to hardcode intermediate base
            classes)
        :param exclude_private: If true, will skip any class that comes from a module
            starting with an underscore (i.e., a private module). This is typically done
            when having shadow duplicate classes implemented in C
        :param exclude_external: If true, will exclude any class that does not originate
            from the same package as the base class.
        :param kwargs: remaining keyword arguments to pass to :func:`Resolver.__init__`

        :returns: A resolver instance
        """
        skip = set(skip) if skip else set()
        return cls(
            {
                subcls
                for subcls in get_subclasses(base, exclude_private=exclude_private, exclude_external=exclude_external)
                if subcls not in skip
            },
            base=base,
            **kwargs,
        )

    def normalize_inst(self, x: X) -> str:
        """Normalize the class name of the instance."""
        return self.normalize_cls(x.__class__)

    def normalize_cls(self, cls: type[X]) -> str:
        """Normalize the class name."""
        return self.normalize(cls.__name__)

    def lookup(self, query: HintOrType[X], default: type[X] | None = None) -> type[X]:
        """Lookup a class."""
        return get_cls(
            query,
            base=self.base,
            lookup_dict=self.lookup_dict,
            lookup_dict_synonyms=self.synonyms,
            default=default or self.default,
            suffix=self.suffix,
        )

    def signature(self, query: HintOrType[X]) -> inspect.Signature:
        """Get the signature for the given class via :func:`inspect.signature`."""
        cls = self.lookup(query)
        return inspect.signature(cls)

    def supports_argument(self, query: HintOrType[X], parameter_name: str) -> bool:
        """Determine if the class constructor supports the given argument."""
        return parameter_name in self.signature(query).parameters

    def make(
        self,
        query: HintOrType[X],
        pos_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> X:
        """Instantiate a class with optional kwargs."""
        if query is None or isinstance(query, str | type):
            cls: type[X] = self.lookup(query)
            try:
                return cls(**(pos_kwargs or {}), **kwargs)
            except TypeError as e:
                if "required keyword-only argument" in e.args[0]:
                    raise KeywordArgumentError(cls, e.args[0]) from None
                if any(text in e.args[0] for text in MISSING_ARGS):
                    raise UnexpectedKeywordError(cls) from None
                raise e

        # An instance was passed, and it will go through without modification.
        return query

    def make_from_kwargs(
        self,
        data: Mapping[str, Any],
        key: str,
        *,
        kwargs_suffix: str = "kwargs",
        **o_kwargs: Any,
    ) -> X:
        """Instantiate a class, by looking up query/pos_kwargs from a dictionary.

        :param data: A dictionary that contains entry ``key`` and entry
            ``{key}_{kwargs_suffix}``.
        :param key: The key in the dictionary whose value will be put in the ``query``
            argument of :func:`make`.
        :param kwargs_suffix: The suffix after ``key`` to look up the data. For example,
            if ``key='model'`` and ``kwargs_suffix='kwargs'`` (the default value), then
            the kwargs from :func:`make` are looked up via ``data['model_kwargs']``.
        :param o_kwargs: Additional kwargs to be passed to :func:`make`

        :returns: An instance of the X datatype parametrized by this resolver
        """
        query = data.get(key, None)
        pos_kwargs = data.get(f"{key}_{kwargs_suffix}", {})
        return self.make(query=query, pos_kwargs=pos_kwargs, **o_kwargs)

    def make_many(
        self,
        queries: OneOrManyHintOrType[X] = None,
        kwargs: OneOrManyOptionalKwargs = None,
        **common_kwargs: Any,
    ) -> list[X]:
        """Resolve and compose several queries together.

        :param queries: One of the following:

            1. none (will result in the default X),
            2. a single X, as either a class, instance, or string for class name
            3. a sequence of X's, as either a class, instance, or string for class name
        :param kwargs: Either none (will use all defaults), a single dictionary (will be
            used for all instances), or a list of dictionaries with the same length as
            ``queries``
        :param common_kwargs: additional keyword-based parameters passed to all
            instantiated instances.

        :returns: A list of X instances

        :raises ValueError: If the number of queries and kwargs has a mismatch
        """
        _query_list: Sequence[HintType[X]]
        _kwargs_list: Sequence[Mapping[str, Any] | None]

        # Prepare the query list
        if queries is not None:
            # FIXME, on first pass i think this should work. needs rethinking
            _query_list = upgrade_to_sequence(queries)  # type:ignore
        elif self.default is None:
            raise ValueError
        else:
            _query_list = [self.default]

        # Prepare the keyword arguments list
        if kwargs is None:
            _kwargs_list = [None] * len(_query_list)
        else:
            _kwargs_list = upgrade_to_sequence(kwargs)

        if 1 == len(_query_list) and 1 < len(_kwargs_list):
            _query_list = list(_query_list) * len(_kwargs_list)
        if 0 < len(_kwargs_list) and 0 == len(_query_list):
            raise ValueError("Keyword arguments were given but no query")
        elif 1 == len(_kwargs_list) == 1 and 1 < len(_query_list):
            _kwargs_list = list(_kwargs_list) * len(_query_list)
        elif len(_kwargs_list) != len(_query_list):
            raise ValueError("Mismatch in number number of queries and kwargs")
        return [
            self.make(query=_result_tracker, pos_kwargs=_result_tracker_kwargs, **common_kwargs)
            for _result_tracker, _result_tracker_kwargs in zip(_query_list, _kwargs_list, strict=False)
        ]

    def make_table(
        self,
        key_fmt: str = "``{key}``",
        cls_fmt: str = ":class:`~{cls}`",
        header: tuple[str, str] = ("key", "class"),
        table_fmt: str = "rst",
        **kwargs: Any,
    ) -> str:
        """Render the table of options in a format suitable for Sphinx documentation.

        :param key_fmt: A format string with a placeholder ``key`` which is filled with
            the normalized key for the class
        :param cls_fmt: A format string with a place-holder ``cls`` which is filled by
            the fully qualified import name.
        :param header: The header of the table.
        :param table_fmt: The table format; passed to :func:`tabulate.tabulate`.
        :param kwargs: Additional keyword-based parameters passed to
            :func:`tabulate.tabulate`.

        :returns: A string containing the formatted table.
        """
        import tabulate

        # TODO: synonyms?
        rows = [
            (key_fmt.format(key=norm_key), cls_fmt.format(cls=f"{cls.__module__}.{cls.__qualname__}"))
            for norm_key, cls in self.lookup_dict.items()
        ]
        return tabulate.tabulate(rows, headers=header, tablefmt=table_fmt, **kwargs)


#: An alias to ClassResolver for backwards compatibility
Resolver = ClassResolver


def get_cls(
    query: HintOrType[X],
    base: type[X],
    lookup_dict: Mapping[str, type[X]],
    lookup_dict_synonyms: Mapping[str, type[X]] | None = None,
    default: type[X] | None = None,
    suffix: str | None = None,
) -> type[X]:
    """Get a class by string, default, or implementation."""
    if query is None:
        if default is None:
            raise ValueError(f"No default {base.__name__} set")
        return default
    elif not isinstance(query, str | type | base):
        raise TypeError(f"Invalid {base.__name__} type: {type(query)} - {query}")
    elif isinstance(query, str):
        key = normalize_string(query, suffix=suffix)
        if key in lookup_dict:
            return lookup_dict[key]
        elif lookup_dict_synonyms is not None and key in lookup_dict_synonyms:
            return lookup_dict_synonyms[key]
        else:
            valid_choices = sorted(set(lookup_dict.keys()).union(lookup_dict_synonyms or []))
            raise KeyError(
                f"Invalid {base.__name__} name: {query} (normalized to: {key}). Valid choices are: {valid_choices}"
            )
    elif isinstance(query, base):
        return query.__class__
    elif isinstance(query, type) and issubclass(query, base):
        return query
    raise TypeError(f"Not subclass of {base.__name__}: {query}")
