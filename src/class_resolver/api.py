# -*- coding: utf-8 -*-

"""Resolve classes."""

import inspect
from operator import attrgetter
from textwrap import dedent
from typing import (
    Any, Callable, Collection, Dict, Generic, Iterable, Iterator, Mapping, Optional, Set, TYPE_CHECKING, Type,
    TypeVar, Union,
)

if TYPE_CHECKING:
    import click

__all__ = [
    'InstOrType',
    'Lookup',
    'LookupType',
    'LookupOrType',
    'Hint',
    'HintType',
    'HintOrType',
    'Resolver',
    'get_subclasses',
    'get_cls',
    'normalize_string',
]

X = TypeVar('X')
Y = TypeVar('Y')

InstOrType = Union[X, Type[X]]

Lookup = Union[str, X]
LookupType = Lookup[Type[X]]
LookupOrType = Lookup[InstOrType[X]]

Hint = Optional[Lookup[X]]
HintType = Hint[Type[X]]
HintOrType = Hint[InstOrType[X]]


class Resolver(Generic[X]):
    """Resolve from a list of classes."""

    #: The base class
    base: Type[X]
    #: The shared suffix fo all classes derived from the base class
    suffix: str
    #: The mapping from normalized class names to the classes indexed by this resolver
    lookup_dict: Dict[str, Type[X]]
    #: The mapping from synonyms to the classes indexed by this resolver
    synonyms: Dict[str, Type[X]]

    def __init__(
        self,
        classes: Collection[Type[X]],
        *,
        base: Type[X],
        default: Optional[Type[X]] = None,
        suffix: Optional[str] = None,
        synonyms: Optional[Mapping[str, Type[X]]] = None,
    ) -> None:
        """Initialize the resolver.

        :param classes: A list of classes
        :param base: The base class
        :param default: The default class
        :param suffix: The optional shared suffix of all classes. If None, use the base class' name for it. To disable
            this behaviour, explicitly provide `suffix=""`.
        :param synonyms: The optional synonym dictionary
        """
        self.base = base
        self.default = default
        if suffix is None:
            suffix = normalize_string(base.__name__)
        self.suffix = suffix
        self.synonyms = dict(synonyms or {})
        self.lookup_dict = {}
        for cls in classes:
            self.register(cls)

    def register(self, cls: Type[X], synonyms: Optional[Iterable[str]] = None, raise_on_conflict: bool = True) -> None:
        """Register an additional class with this resolver.

        :param cls: The class to register
        :param synonyms: An optional iterable of synonyms to add for the class
        :param raise_on_conflict: Determines the behavior when a conflict is encountered on either
            the normalized class name or a synonym. If true, will raise an exception. If false, will
            simply disregard the entry.

        :raises KeyError: If ``raise_on_conflict`` is true and there's a conflict in either the class
            name or a synonym name.
        """
        key = self.normalize_cls(cls)
        if key in self.lookup_dict:
            if raise_on_conflict:
                raise KeyError(f'This resolver already contains a class with key {key}: {self.lookup_dict[key]}')
            else:
                return

        self.lookup_dict[key] = cls
        for synonym in synonyms or []:
            if synonym in self.synonyms:
                if raise_on_conflict:
                    raise KeyError(f'This resolver already contains synonym {synonym} for {self.synonyms[synonym]}')
                else:
                    continue
            self.synonyms[synonym] = cls

    def __iter__(self) -> Iterator[Type[X]]:
        """Return an iterator over the indexed classes sorted by name."""
        return iter(sorted(self.lookup_dict.values(), key=attrgetter('__name__')))

    @classmethod
    def from_subclasses(cls, base: Type[X], *, skip: Optional[Collection[Type[X]]] = None, **kwargs) -> 'Resolver':
        """Make a resolver from the subclasses of a given class.

        :param base: The base class whose subclasses will be indexed
        :param skip: Any subclasses to skip (usually good to hardcode intermediate base classes)
        :param kwargs: remaining keyword arguments to pass to :func:`Resolver.__init__`
        :return: A resolver instance
        """
        skip = set(skip) if skip else set()
        return Resolver(
            {
                subcls
                for subcls in get_subclasses(base)
                if subcls not in skip
            },
            base=base,
            **kwargs,
        )

    def normalize_inst(self, x: X) -> str:
        """Normalize the class name of the instance."""
        return self.normalize_cls(x.__class__)

    def normalize_cls(self, cls: Type[X]) -> str:
        """Normalize the class name."""
        return self.normalize(cls.__name__)

    def normalize(self, s: str) -> str:
        """Normalize the string with this resolve's suffix."""
        return normalize_string(s, suffix=self.suffix)

    def lookup(self, query: Hint[Type[X]]) -> Type[X]:
        """Lookup a class."""
        return get_cls(
            query,
            base=self.base,
            lookup_dict=self.lookup_dict,
            lookup_dict_synonyms=self.synonyms,
            default=self.default,
            suffix=self.suffix,
        )

    def signature(self, query: Hint[Type[X]]) -> inspect.Signature:
        """Get the signature for the given class via :func:`inspect.signature`."""
        cls = self.lookup(query)
        return inspect.signature(cls)

    def supports_argument(self, query: Hint[Type[X]], parameter_name: str) -> bool:
        """Determine if the class constructor supports the given argument."""
        return parameter_name in self.signature(query).parameters

    def make(self, query: HintOrType[X], pos_kwargs: Optional[Mapping[str, Any]] = None, **kwargs) -> X:
        """Instantiate a class with optional kwargs."""
        if query is None or isinstance(query, (str, type)):
            cls: Type[X] = self.lookup(query)
            return cls(**(pos_kwargs or {}), **kwargs)  # type: ignore

        # An instance was passed, and it will go through without modification.
        return query

    def make_from_kwargs(
        self,
        data: Mapping[str, Any],
        key: str,
        *,
        kwargs_suffix: str = "kwargs",
        **o_kwargs,
    ) -> X:
        """Instantiate a class, by looking up query/pos_kwargs from a dictionary.

        :param data: A dictionary that contains entry ``key`` and entry ``{key}_{kwargs_suffix}``.
        :param key: The key in the dictionary whose value will be put in the ``query`` argument of :func:`make`.
        :param kwargs_suffix: The suffix after ``key`` to look up the data. For example, if ``key='model'``
            and ``kwargs_suffix='kwargs'`` (the default value), then the kwargs from :func:`make` are looked up
            via ``data['model_kwargs']``.
        :param o_kwargs: Additional kwargs to be passed to :func:`make`
        :returns: An instance of the X datatype parametrized by this resolver
        """
        query = data.get(key, None)
        pos_kwargs = data.get(f"{key}_{kwargs_suffix}", {})
        return self.make(query=query, pos_kwargs=pos_kwargs, **o_kwargs)

    def get_option(self, *flags: str, default: Hint[Type[X]] = None, as_string: bool = False, **kwargs):
        """Get a click option for this resolver."""
        if default is None:
            if self.default is None:
                raise ValueError('no default given either from resolver or explicitly')
            default = self.default
        else:
            default = self.lookup(default)
        default = self.normalize_cls(default)

        import click

        return click.option(
            *flags,
            type=click.Choice(list(self.lookup_dict), case_sensitive=False),
            default=default,
            show_default=True,
            callback=None if as_string else _make_callback(self.lookup),
            **kwargs,
        )

    @property
    def options(self) -> Set[str]:
        """Return the normalized option names."""
        return set(self.lookup_dict.keys())

    @property
    def classes(self) -> Set[Type[X]]:
        """Return the available classes."""
        return set(self.lookup_dict.values())

    def ray_tune_search_space(self, kwargs_search_space: Optional[Mapping[str, Any]] = None):
        """Return a search space for ray.tune.

        ray.tune is a package for distributed hyperparameter optimization. The search space for this search is defined
        as a (nested) dictionary, which can contain special values `tune.{choice,uniform,...}`. For these values, the
        search algorithm will sample a specific configuration.

        This method can be used to create a tune.choice sampler for the choices available to the resolver. By default,
        this is equivalent to

        .. code-block:: python

            ray.tune.choice(self.options)

        If additional `kwargs_search_space` are passed, they are assumed to be a sub-search space for the constructor
        parameters passed via `pos_kwargs`.  The resulting sub-search thus looks as follows:

        .. code-block:: python

            ray.tune.choice(
                query=self.options,
                **kwargs_search_space,
            )

        :param kwargs_search_space:
            Additional sub search space for the constructor's parameters.

        :return:
            A ray.tune compatible search space.

        :raises ImportError:
            If ray.tune is not installed.

        .. seealso ::
            https://docs.ray.io/en/master/tune/index.html
        """
        try:
            import ray.tune
        except ImportError:
            raise ImportError(dedent(
                """
                To use ray_tune_search_space please install ray tune first.

                You can do so by selecting the appropriate install option for the package

                    pip install class-resolver[ray]

                or by manually installing ray tune

                    pip install ray[tune]
                """,
            )) from None

        query = ray.tune.choice(self.options)

        if kwargs_search_space is None:
            return query

        return dict(
            query=query,
            **kwargs_search_space,
        )


def _not_hint(x: Any) -> bool:
    return x is not None and not isinstance(x, (str, type))


def get_subclasses(cls: Type[X]) -> Iterable[Type[X]]:
    """Get all subclasses.

    :param cls: The ancestor class
    :yields: Descendant classes of the ancestor class
    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def get_cls(
    query: Union[None, str, Type[X]],
    base: Type[X],
    lookup_dict: Mapping[str, Type[X]],
    lookup_dict_synonyms: Optional[Mapping[str, Type[X]]] = None,
    default: Optional[Type[X]] = None,
    suffix: Optional[str] = None,
) -> Type[X]:
    """Get a class by string, default, or implementation."""
    if query is None:
        if default is None:
            raise ValueError(f'No default {base.__name__} set')
        return default
    elif not isinstance(query, (str, type)):
        raise TypeError(f'Invalid {base.__name__} type: {type(query)} - {query}')
    elif isinstance(query, str):
        key = normalize_string(query, suffix=suffix)
        if key in lookup_dict:
            return lookup_dict[key]
        if lookup_dict_synonyms is not None and key in lookup_dict_synonyms:
            return lookup_dict_synonyms[key]
        valid_choices = sorted(set(lookup_dict.keys()).union(lookup_dict_synonyms or []))
        raise ValueError(f'Invalid {base.__name__} name: {query}. Valid choices are: {valid_choices}')
    elif issubclass(query, base):
        return query
    raise TypeError(f'Not subclass of {base.__name__}: {query}')


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace('-', '').replace('_', '').replace(' ', '')
    if suffix is not None and s.endswith(suffix.lower()):
        return s[:-len(suffix)]
    return s


def _make_callback(f: Callable[[X], Y]) -> Callable[['click.Context', 'click.Parameter', X], Y]:
    def _callback(_ctx: 'click.Context', _param: 'click.Parameter', value: X) -> Y:
        return f(value)

    return _callback
