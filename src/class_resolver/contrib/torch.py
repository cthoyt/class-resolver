# -*- coding: utf-8 -*-

"""Class resolvers for PyTorch."""

from torch.optim import Adam, Optimizer

from ..api import Resolver

optimizer_resolver = Resolver.from_subclasses(Optimizer, default=Adam)

if __name__ == '__main__':
    print(optimizer_resolver.options)
