"""Utilities for automatic documentation."""

from __future__ import annotations

import inspect
import textwrap
from typing import Callable, TypeVar

__all__ = [
    "document_resolver",
]

F = TypeVar("F", bound=Callable)


def document_resolver(
    *params: str | tuple[str, str],
    resolver_name: str,
) -> Callable[[F], F]:
    """
    Build a decorator to add information about resolved parameter pairs.

    The decorator is intended for methods with follow the ``param`` + ``param_kwargs`` pattern and internally use a
    class resolver.

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver
        from class_resolver.contrib.torch import activation_resolver

        @document_resolver(
            "activation", resolver_name="class_resolver.contrib.torch.activation_resolver"
        )
        def f(
            tensor: Tensor,
            activation: Union[None, str, type[nn.Module], nn.Module],
            activation_kwargs: dict[str, Any] | None,
        ):
            _activation = activation_resolver.make(activation, activation_kwargs)
            return _activation(tensor)


    This also can be stacked for multiple resolvers.

    .. code-block:: python

        from typing import Any, Union
        from torch import Tensor, nn
        from class_resolver import document_resolver
        from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

        @document_resolver(
            "activation", resolver_name="class_resolver.contrib.torch.activation_resolver"
        )
        @document_resolver(
            "aggregation", resolver_name="class_resolver.contrib.torch.aggregation_resolver"
        )
        def f(
            tensor: Tensor,
            activation: Union[None, str, type[nn.Module], nn.Module],
            activation_kwargs: dict[str, Any] | None,
            aggregation: Union[None, str, type[nn.Module], nn.Module],
            aggregation_kwargs: dict[str, Any] | None,
        ):
            _activation = activation_resolver.make(activation, activation_kwargs)
            _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
            return _aggregation(_activation(tensor))

    :param params:
        the name of the parameters. Will be automatically completed to include all the ``_kwargs`` suffixed parts, too.
    :param resolver_name:
        the fully qualified path of the resolver used to construct a reference via the ``:data:`` role.

    :return:
        a decorator which extends a function's docstring.
    :raises ValueError:
        When either no parameter name was provided, there was a duplicate parameter name.
    """
    # input validation
    if not params:
        raise ValueError("Must provided at least one parameter name.")

    # normalize parameter name pairs
    param_pairs: list[tuple[str, str]] = []
    for pair in params:
        if isinstance(pair, str):
            pair = (pair, f"{pair}_kwargs")
        elif len(pair) != 2:
            raise ValueError(f"Invalid parameter pair {pair}")
        param_pairs.append(pair)

    # check for duplicates
    expanded_params = set(e for pair in param_pairs for e in pair)
    if len(expanded_params) < 2 * len(param_pairs):
        raise ValueError(f"There are duplicates in (the expanded) {params=}")

    # TODO: we could do some more sanitization, e.g., importing the resolver, trying to match types, ...

    def add_note(func: F) -> F:
        """
        Extend the function's docstring with a note about resolved parameters.

        :param func:
            the function to decorate.

        :return:
            the function with extended docstring.

        :raises ValueError:
            When the signature does not contain the resolved parameter names, or the docstring is missing.
        """
        signature = inspect.signature(func)
        if missing := expanded_params.difference(signature.parameters):
            raise ValueError(f"{missing=} parameters in {signature=}.")
        if not func.__doc__:
            raise ValueError("docstring is empty")
        pairs_str = ", ".join(f"``({param}, {param_kwargs})``" for param, param_kwargs in param_pairs)
        note_str = textwrap.dedent(
            f"""\
            .. note ::

                The parameter pairs {pairs_str} are passed to :data:`{resolver_name}`.
                An explanation of resolvers and how to use them is given in
                https://class-resolver.readthedocs.io/en/latest/.
            """
        )
        note_str = textwrap.indent(text=note_str, prefix="        ", predicate=bool)
        # TODO: this is in-place
        func.__doc__ = f"{func.__doc__.lstrip()}\n\n{note_str}"
        return func

    return add_note
