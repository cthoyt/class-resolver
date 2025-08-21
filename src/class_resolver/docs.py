"""Utilities for automatic documentation."""

from __future__ import annotations

import importlib
import inspect
import textwrap
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from .base import BaseResolver

__all__ = [
    "ResolverKey",
    "update_docstring_with_resolver_keys",
]

T = TypeVar("T")
P = ParamSpec("P")


def _get_qualpath_from_object(resolver: BaseResolver[Any, Any]) -> str:
    if resolver.location:
        return resolver.location
    raise NotImplementedError(
        "Can not get a qualified name for auto-generation of sphinx documentation "
        "for a resolver that doesn't have the `location` variable set"
    )


class ResolverKey:
    """An object storing information about how a resolver is used in a signature."""

    name: str
    key: str
    resolver_path: str
    # note that resolver keys don't depend at all on the
    # types in the resolver
    resolver: BaseResolver[Any, Any] | None

    def __init__(
        self,
        name: str,
        resolver: str | BaseResolver[Any, Any],
        key: str | None = None,
    ) -> None:
        """Initialize the key for :func:`update_docstring_with_resolver_keys`."""
        self.name = name
        self.key = f"{self.name}_kwargs" if key is None else key

        if isinstance(resolver, str):
            self.resolver_path = resolver
            try:
                module_name, variable_name = resolver.rsplit(".", 1)
                module = importlib.import_module(module_name)
                resolver_inst = getattr(module, variable_name)
            except (ImportError, ValueError):
                self.resolver = None
            except AttributeError as e:
                if "partially initialized module" not in str(e):
                    raise
                # this happens in a circular import case. just let it go
                self.resolver = None
            else:
                self.resolver = resolver_inst
        elif isinstance(resolver, BaseResolver):
            self.resolver_path = _get_qualpath_from_object(resolver)
            self.resolver = resolver
        else:
            raise TypeError


def _clean_docstring(s: str) -> str:
    """Clean a docstring.

    :param s: Input docstring

    :returns: Cleaned docstring

    :raises ValueError: if the docstring is improperly formatted

    This method does the following

    1. strip
    2. pop off first line
    3. ensure second line is blank
    4. dedent on all remaining lines
    5. chunk em back together
    """
    s = s.strip()
    lines = s.splitlines()
    if len(lines) == 1:
        return lines[0]
    if len(lines) == 2:
        if not lines[1].strip():
            return lines[0]
        else:
            raise ValueError("not sure how to clean a two line docstring")
    first, second, *rest = lines
    if second.strip():
        raise ValueError
    rest_j = "\n".join(rest)
    rest_j = textwrap.dedent(rest_j)
    return f"{first.strip()}\n\n{rest_j}"


def update_docstring_with_resolver_keys(*resolver_keys: ResolverKey) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Build a decorator to add information about resolved parameter pairs.

    The decorator is intended for methods with follow the ``param`` + ``param_kwargs``
    pattern and internally use a class resolver.

    .. code-block:: python

        from typing import Any
        from torch import Tensor, nn
        from class_resolver import update_docstring_with_resolver_keys, ResolverKey
        from class_resolver.contrib.torch import activation_resolver


        @update_docstring_with_resolver_keys(
            ResolverKey("activation", "class_resolver.contrib.torch.activation_resolver"),
        )
        def f(
            tensor: Tensor,
            activation: None | str | type[nn.Module] | nn.Module,
            activation_kwargs: dict[str, Any] | None,
        ) -> Tensor:
            _activation = activation_resolver.make(activation, activation_kwargs)
            return _activation(tensor)

    This also can be stacked for multiple resolvers.

    .. code-block:: python

        from typing import Any
        from torch import Tensor, nn
        from class_resolver import update_docstring_with_resolver_keys
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver


        @update_docstring_with_resolver_keys(
            ResolverKey("activation", "class_resolver.contrib.torch.activation_resolver"),
            ResolverKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
        )
        def f(
            tensor: Tensor,
            activation: None | str | type[nn.Module] | nn.Module,
            activation_kwargs: dict[str, Any] | None,
            aggregation: None | str | type[nn.Module] | nn.Module,
            aggregation_kwargs: dict[str, Any] | None,
        ) -> Tensor:
            _activation = activation_resolver.make(activation, activation_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _aggregation(_activation(tensor))

    It might be the case that you have two different arguments that use the same
    resolver. No prob!

    .. code-block:: python

        from typing import Any
        from torch import Tensor, nn
        from class_resolver import update_docstring_with_resolver_keys
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver


        @update_docstring_with_resolver_keys(
            ResolverKey("activation_1", "class_resolver.contrib.torch.activation_resolver"),
            ResolverKey("activation_2", "class_resolver.contrib.torch.activation_resolver"),
            ResolverKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
        )
        def f(
            tensor: Tensor,
            activation_1: None | str | type[nn.Module] | nn.Module,
            activation_1_kwargs: dict[str, Any] | None,
            aggregation: None | str | type[nn.Module] | nn.Module,
            aggregation_kwargs: dict[str, Any] | None,
            activation_2: None | str | type[nn.Module] | nn.Module,
            activation_2_kwargs: dict[str, Any] | None,
        ) -> Tensor:
            _activation_1 = activation_resolver.make(activation_1, activation_1_kwargs)
            _activation_2 = activation_resolver.make(activation_2, activation_2_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _activation_2(_aggregation(_activation_2(tensor)))

    :param resolver_keys: A variadic list of keys, each describing:

        1. the names of the parameter
        2. the resolver used to construct a reference via the ``:data:`` role.
        3. the name of the parameter for giving keyword arguments. By default, this is
           constructed by taking the name and post-pending ``_kwargs``.

    :returns: a decorator which extends a function's docstring.

    :raises ValueError: When either no parameter name was provided, there was a
        duplicate parameter name.
    """
    # input validation
    if not resolver_keys:
        raise ValueError("Must provided at least one parameter name.")

    # check for duplicates
    expanded_params = {e for key in resolver_keys for e in (key.name, key.key)}
    if len(expanded_params) < 2 * len(resolver_keys):
        raise ValueError(f"There are duplicates in (the expanded) {resolver_keys=}")

    # TODO: we could do some more sanitization, e.g., trying to match types, ...

    def add_note(func: Callable[P, T]) -> Callable[P, T]:
        """Extend the function's docstring with a note about resolved parameters.

        :param func: the function to decorate.

        :returns: the function with extended docstring.

        :raises ValueError: When the signature does not contain the resolved parameter
            names, or the docstring is missing.
        """
        signature = inspect.signature(func)
        if missing := expanded_params.difference(signature.parameters):
            raise ValueError(f"{missing=} parameters in {signature=}.")
        if not func.__doc__:
            raise ValueError("docstring is empty")

        resolver_to_keys = defaultdict(list)
        for key in resolver_keys:
            resolver_to_keys[key.resolver_path].append(key)

        parameter_pair_strs = []
        for resolver_qualname, subkeys in resolver_to_keys.items():
            pair_strs = [f"``({key.name}, {key.key})``" for key in subkeys]
            if len(subkeys) > 1:
                parameter_pair_str = f"pairs {', '.join(pair_strs)} are"
            else:
                parameter_pair_str = f"pair {pair_strs[0]} is"
            parameter_pair_strs.append(f"The parameter {parameter_pair_str} used for :data:`{resolver_qualname}`")

        if len(parameter_pair_strs) == 1:
            note_str = f"""\
            .. note ::

                {parameter_pair_strs[0]}

                An explanation of resolvers and how to use them is given in
                https://class-resolver.readthedocs.io/en/latest/.
            """
        else:
            bullet_points = "\n".join(" " * 4 + "- " + i for i in parameter_pair_strs)
            note_str = f"""\
.. note ::

    {len(resolver_keys)} resolvers are used in this function.

{bullet_points}

    An explanation of resolvers and how to use them is given in
    https://class-resolver.readthedocs.io/en/latest/.
"""
        func.__doc__ = f"{_clean_docstring(func.__doc__)}\n\n{textwrap.dedent(note_str)}".rstrip()
        return func

    return add_note
