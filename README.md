<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  Class Resolver
</h1>

<p align="center">
    <a href="https://github.com/cthoyt/class-resolver/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/cthoyt/class-resolver/workflows/Tests/badge.svg" />
    </a>
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href="https://pypi.org/project/class_resolver">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/class_resolver" />
    </a>
    <a href="https://pypi.org/project/class_resolver">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/class_resolver" />
    </a>
    <a href="https://github.com/cthoyt/class-resolver/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/class-resolver" />
    </a>
    <a href='https://class_resolver.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/class_resolver/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/cthoyt/class-resolver/branch/main">
        <img src="https://codecov.io/gh/cthoyt/class-resolver/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://zenodo.org/badge/latestdoi/343741010">
        <img src="https://zenodo.org/badge/343741010.svg" alt="DOI">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</p>

Lookup and instantiate classes with style.

## 💪 Getting Started

```python
from class_resolver import ClassResolver
from dataclasses import dataclass

class Base: pass

@dataclass
class A(Base):
   name: str

@dataclass
class B(Base):
   name: str

# Index
resolver = ClassResolver([A, B], base=Base)

# Lookup
assert A == resolver.lookup('A')

# Instantiate with a dictionary
assert A(name='hi') == resolver.make('A', {'name': 'hi'})

# Instantiate with kwargs
assert A(name='hi') == resolver.make('A', name='hi')

# A pre-instantiated class will simply be passed through
assert A(name='hi') == resolver.make(A(name='hi'))
```

## 🤖 Writing Extensible Machine Learning Models with `class-resolver`

Assume you've implemented a simple multi-layer perceptron in PyTorch:

```python
from itertools import chain

from more_itertools import pairwise
from torch import nn

class MLP(nn.Sequential):
    def __init__(self, dims: list[int]):
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            )
            for in_features, out_features in pairwise(dims)
        ))
```

This MLP uses a hard-coded rectified linear unit as the non-linear activation
function between layers. We can generalize this MLP to use a variety of
non-linear activation functions by adding an argument to its
`__init__()` function like in:

```python
from itertools import chain

from more_itertools import pairwise
from torch import nn

class MLP(nn.Sequential):
    def __init__(self, dims: list[int], activation: str = "relu"):
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "tanh":
            activation = nn.Tanh()
        elif activation == "hardtanh":
            activation = nn.Hardtanh()
        else:
            raise KeyError(f"Unsupported activation: {activation}")
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation,
            )
            for in_features, out_features in pairwise(dims)
        ))
```

The first issue with this implementation is it relies on a hard-coded set of
conditional statements and is therefore hard to extend. It can be improved
by using a dictionary lookup:

```python
from itertools import chain

from more_itertools import pairwise
from torch import nn

activation_lookup: dict[str, nn.Module] = {
   "relu": nn.ReLU(),
   "tanh": nn.Tanh(),
   "hardtanh": nn.Hardtanh(),
}

class MLP(nn.Sequential):
    def __init__(self, dims: list[int], activation: str = "relu"):
        activation = activation_lookup[activation]
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation,
            )
            for in_features, out_features in pairwise(dims)
        ))
```

This approach is rigid because it requires pre-instantiation of the activations.
If we needed to vary the arguments to the `nn.HardTanh` class, the previous
approach wouldn't work. We can change the implementation to lookup on the 
class *before instantiation* then optionally pass some arguments:

```python
from itertools import chain

from more_itertools import pairwise
from torch import nn

activation_lookup: dict[str, type[nn.Module]] = {
   "relu": nn.ReLU,
   "tanh": nn.Tanh,
   "hardtanh": nn.Hardtanh,
}

class MLP(nn.Sequential):
    def __init__(
        self, 
        dims: list[int], 
        activation: str = "relu", 
        activation_kwargs: None | dict[str, any] = None,
    ):
        activation_cls = activation_lookup[activation]
        activation = activation_cls(**(activation_kwargs or {}))
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation,
            )
            for in_features, out_features in pairwise(dims)
        ))
```

This is pretty good, but it still has a few issues:
1. you have to manually maintain the `activation_lookup` dictionary,
2. you can't pass an instance or class through the `activation` keyword
3. you have to get the casing just right
4. the default is hard-coded as a string, which means this has to get copied
   (error-prone) in any place that creates an MLP
5. you have to re-write this logic for all of your classes

Enter the `class_resolver` package, which takes care of all of these
things using the following:

```python
from itertools import chain

from class_resolver import ClassResolver, Hint
from more_itertools import pairwise
from torch import nn

activation_resolver = ClassResolver(
    [nn.ReLU, nn.Tanh, nn.Hardtanh],
    base=nn.Module,
    default=nn.ReLU,
)

class MLP(nn.Sequential):
    def __init__(
        self, 
        dims: list[int], 
        activation: Hint[nn.Module] = None,  # Hint = Union[None, str, nn.Module, type[nn.Module]]
        activation_kwargs: None | dict[str, any] = None,
    ):
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation_resolver.make(activation, activation_kwargs),
            )
            for in_features, out_features in pairwise(dims)
        ))
```

Because this is such a common pattern, we've made it available through contrib
module in `class_resolver.contrib.torch`:

```python
from itertools import chain

from class_resolver import Hint
from class_resolver.contrib.torch import activation_resolver
from more_itertools import pairwise
from torch import nn

class MLP(nn.Sequential):
    def __init__(
        self, 
        dims: list[int], 
        activation: Hint[nn.Module] = None,
        activation_kwargs: None | dict[str, any] = None,
    ):
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation_resolver.make(activation, activation_kwargs),
            )
            for in_features, out_features in pairwise(dims)
        ))
```

Now, you can instantiate the MLP with any of the following:

```python
MLP(dims=[10, 200, 40])  # uses default, which is ReLU
MLP(dims=[10, 200, 40], activation="relu")  # uses lowercase
MLP(dims=[10, 200, 40], activation="ReLU")  # uses stylized
MLP(dims=[10, 200, 40], activation=nn.ReLU)  # uses class
MLP(dims=[10, 200, 40], activation=nn.ReLU())  # uses instance

MLP(dims=[10, 200, 40], activation="hardtanh", activation_kwargs={"min_val": 0.0, "max_value": 6.0})  # uses kwargs
MLP(dims=[10, 200, 40], activation=nn.HardTanh, activation_kwargs={"min_val": 0.0, "max_value": 6.0})  # uses kwargs
MLP(dims=[10, 200, 40], activation=nn.HardTanh(0.0, 6.0))  # uses instance
```

In practice, it makes sense to stick to using the strings in combination with
hyper-parameter optimization libraries like [Optuna](https://optuna.org/).

## ⬇️ Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/class_resolver/) with:

```bash
$ pip install class_resolver
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/cthoyt/class-resolver.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/cthoyt/class-resolver.git
$ cd class-resolver
$ pip install -e .
```

## 🙏 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/cthoyt/class-resolver/blob/master/CONTRIBUTING.rst) for more
information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/{{cookiecutter.github_organization_name}}/{{cookiecutter.github_repository_name}}/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/{{cookiecutter.package_name}}/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
