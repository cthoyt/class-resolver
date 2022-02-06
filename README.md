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
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-python--package-yellow" /> 
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
from class_resolver import Resolver
from dataclasses import dataclass

class Base: pass

@dataclass
class A(Base):
   name: str

@dataclass
class B(Base):
   name: str

# Index
resolver = Resolver([A, B], base=Base)

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

This architecture can be used to describe an entire class of MLPs with
different activation functions, so it might make sense to generalize:

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

A better way would be to create a dictionary:

```python
from itertools import chain

from more_itertools import pairwise
from torch import nn

activations: dict[str, type[nn.Module]] = {
   "relu": nn.ReLU,
   "tanh": nn.Tanh,
}

class MLP(nn.Sequential):
    def __init__(self, dims: list[int], activation: str = "relu"):
        activation_cls = activations[activation]
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation_cls(),
            )
            for in_features, out_features in pairwise(dims)
        ))
```

The `class-resolver` takes care of this, and more:

```python
from torch import nn
from itertools import chain
from more_itertools import pairwise
from class_resolver import ClassResolver, Hint

activation_resolver = ClassResolver(
    [nn.ReLU, nn.Tanh, nn.Sigmoid],
    base=nn.Module,
    default=nn.ReLU,
)

class MLP(nn.Sequential):
    def __init__(self, dims: list[int], activation: Hint[str] = None):
        super().__init__(chain.from_iterable(
            (
                nn.Linear(in_features, out_features),
                activation_resolver.make(activation),
            )
            for in_features, out_features in pairwise(dims)
        ))
```

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
