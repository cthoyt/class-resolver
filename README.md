<!--
<p align="center">
  <img src="https://github.com/cthoyt/class-resolver/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  Class Resolver
</h1>

<p align="center">
    <a href="https://github.com/cthoyt/class-resolver/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/cthoyt/class-resolver/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/class_resolver">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/class_resolver" /></a>
    <a href="https://pypi.org/project/class_resolver">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/class_resolver" /></a>
    <a href="https://github.com/cthoyt/class-resolver/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/class_resolver" /></a>
    <a href='https://class_resolver.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/class_resolver/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://codecov.io/gh/cthoyt/class-resolver/branch/main">
        <img src="https://codecov.io/gh/cthoyt/class-resolver/branch/main/graph/badge.svg" alt="Codecov status" /></a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /></a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href="https://github.com/cthoyt/class-resolver/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/></a>
    <a href="https://zenodo.org/badge/latestdoi/343741010">
        <img src="https://zenodo.org/badge/343741010.svg" alt="DOI"></a>
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
non-linear activation functions by adding an argument to its `__init__()`
function like in:

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
conditional statements and is therefore hard to extend. It can be improved by
using a dictionary lookup:

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
approach wouldn't work. We can change the implementation to lookup on the class
_before instantiation_ then optionally pass some arguments:

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

Enter the `class_resolver` package, which takes care of all of these things
using the following:

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

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/class_resolver/) with uv:

```console
$ uv pip install class_resolver
```

or with pip:

```console
$ python3 -m pip install class_resolver
```

The most recent code and data can be installed directly from GitHub with uv:

```console
$ uv pip install git+https://github.com/cthoyt/class-resolver.git
```

or with pip:

```console
$ python3 -m pip install git+https://github.com/cthoyt/class-resolver.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are
appreciated. See
[CONTRIBUTING.md](https://github.com/cthoyt/class-resolver/blob/master/.github/CONTRIBUTING.md)
for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Biopragmatics Lab](https://biopragmatics.github.io)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body  | Program                                                      | Grant Number |
|---------------|--------------------------------------------------------------|--------------|
| Funder        | [Grant Name (GRANT-ACRONYM)](https://example.com/grant-link) | ABCXYZ       |
-->

### 🍪 Cookiecutter

This package was created with
[@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using
[@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack)
template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a
code contribution.

### Development Installation

To install in development mode, use the following:

```console
$ git clone git+https://github.com/cthoyt/class-resolver.git
$ cd class-resolver
$ uv pip install -e .
```

Alternatively, install using pip:

```console
$ python3 -m pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with
`uv tool install tox --with tox-uv` or `python3 -m pip install tox tox-uv`, the
unit tests in the `tests/` folder can be run reproducibly with:

```console
$ tox -e py
```

Additionally, these tests are automatically re-run with each commit in a
[GitHub Action](https://github.com/cthoyt/class-resolver/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```console
$ git clone git+https://github.com/cthoyt/class-resolver.git
$ cd class-resolver
$ tox -e docs
$ open docs/build/html/index.html
```

The documentation automatically installs the package as well as the `docs` extra
specified in the [`pyproject.toml`](pyproject.toml). `sphinx` plugins like
`texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

The documentation can be deployed to [ReadTheDocs](https://readthedocs.io) using
[this guide](https://docs.readthedocs.io/en/stable/intro/import-guide.html). The
[`.readthedocs.yml`](.readthedocs.yml) YAML file contains all the configuration
you'll need. You can also set up continuous integration on GitHub to check not
only that Sphinx can build the documentation in an isolated environment (i.e.,
with `tox -e docs-test`) but also that
[ReadTheDocs can build it too](https://docs.readthedocs.io/en/stable/pull-requests.html).

</details>

## 🧑‍💻 For Maintainers

<details>
  <summary>See maintainer instructions</summary>

### Initial Configuration

#### Configuring ReadTheDocs

[ReadTheDocs](https://readthedocs.org) is an external documentation hosting
service that integrates with GitHub's CI/CD. Do the following for each
repository:

1. Log in to ReadTheDocs with your GitHub account to install the integration at
   https://readthedocs.org/accounts/login/?next=/dashboard/
2. Import your project by navigating to https://readthedocs.org/dashboard/import
   then clicking the plus icon next to your repository
3. You can rename the repository on the next screen using a more stylized name
   (i.e., with spaces and capital letters)
4. Click next, and you're good to go!

#### Configuring Archival on Zenodo

[Zenodo](https://zenodo.org) is a long-term archival system that assigns a DOI
to each release of your package. Do the following for each repository:

1. Log in to Zenodo via GitHub with this link:
   https://zenodo.org/oauth/login/github/?next=%2F. This brings you to a page
   that lists all of your organizations and asks you to approve installing the
   Zenodo app on GitHub. Click "grant" next to any organizations you want to
   enable the integration for, then click the big green "approve" button. This
   step only needs to be done once.
2. Navigate to https://zenodo.org/account/settings/github/, which lists all of
   your GitHub repositories (both in your username and any organizations you
   enabled). Click the on/off toggle for any relevant repositories. When you
   make a new repository, you'll have to come back to this

After these steps, you're ready to go! After you make "release" on GitHub (steps
for this are below), you can navigate to
https://zenodo.org/account/settings/github/repository/cthoyt/class-resolver to
see the DOI for the release and link to the Zenodo record for it.

#### Registering with the Python Package Index (PyPI)

The [Python Package Index (PyPI)](https://pypi.org) hosts packages so they can
be easily installed with `pip`, `uv`, and equivalent tools.

1. Register for an account [here](https://pypi.org/account/register)
2. Navigate to https://pypi.org/manage/account and make sure you have verified
   your email address. A verification email might not have been sent by default,
   so you might have to click the "options" dropdown next to your address to get
   to the "re-send verification email" button
3. 2-Factor authentication is required for PyPI since the end of 2023 (see this
   [blog post from PyPI](https://blog.pypi.org/posts/2023-05-25-securing-pypi-with-2fa/)).
   This means you have to first issue account recovery codes, then set up
   2-factor authentication
4. Issue an API token from https://pypi.org/manage/account/token

This only needs to be done once per developer.

#### Configuring your machine's connection to PyPI

This needs to be done once per machine.

```console
$ uv tool install keyring
$ keyring set https://upload.pypi.org/legacy/ __token__
$ keyring set https://test.pypi.org/legacy/ __token__
```

Note that this deprecates previous workflows using `.pypirc`.

### 📦 Making a Release

#### Uploading to PyPI

After installing the package in development mode and installing `tox` with
`uv tool install tox --with tox-uv` or `python3 -m pip install tox tox-uv`, run
the following from the console:

```console
$ tox -e finish
```

This script does the following:

1. Uses [bump-my-version](https://github.com/callowayproject/bump-my-version) to
   switch the version number in the `pyproject.toml`, `CITATION.cff`,
   `src/class_resolver/version.py`, and
   [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using
   [`uv build`](https://docs.astral.sh/uv/guides/publish/#building-your-package)
3. Uploads to PyPI using
   [`uv publish`](https://docs.astral.sh/uv/guides/publish/#publishing-your-package).
4. Push to GitHub. You'll need to make a release going with the commit where the
   version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump
   the version by minor, you can use `tox -e bumpversion -- minor` after.

#### Releasing on GitHub

1. Navigate to https://github.com/cthoyt/class-resolver/releases/new to draft a
   new release
2. Click the "Choose a Tag" dropdown and select the tag corresponding to the
   release you just made
3. Click the "Generate Release Notes" button to get a quick outline of recent
   changes. Modify the title and description as you see fit
4. Click the big green "Publish Release" button

This will trigger Zenodo to assign a DOI to your release as well.

### Updating Package Boilerplate

This project uses `cruft` to keep boilerplate (i.e., configuration, contribution
guidelines, documentation configuration) up-to-date with the upstream
cookiecutter package. Install cruft with either `uv tool install cruft` or
`python3 -m pip install cruft` then run:

```console
$ cruft update
```

More info on Cruft's update command is available
[here](https://github.com/cruft/cruft?tab=readme-ov-file#updating-a-project).

</details>
