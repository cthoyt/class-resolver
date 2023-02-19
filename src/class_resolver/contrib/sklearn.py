"""
Scikit-learn is a generic machine learning package with implementations of
algorithms for classification, regression, dimensionality reduction, clustering,
as well as other generic tooling.

The ``class-resolver`` provides several class resolvers for instantiating various
implementations, such as those of linear models.
"""  # noqa:D205,D400

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)

from ..api import ClassResolver

__all__ = [
    "linear_model_resolver",
]

linear_model_resolver: ClassResolver[BaseEstimator] = ClassResolver(
    [
        LogisticRegression,
        LogisticRegressionCV,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        # SGDOneClassSVM,
    ],
    base=BaseEstimator,
    base_as_suffix=False,
    default=LogisticRegression,
)
"""A resolver for linear model classifiers including:

The default value is the class for logistic regressions.
This resolver can be used like in the following:

.. code-block:: python

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from class_resolver.contrib.sklearn import linear_model_resolver

    # Prepare a dataset
    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Lookup with a string
    classifier = linear_model_resolver.make("LogisticRegression")
    classifier.fit(x_train, y_train)
    assert 0.7 < classifier.score(x_test, y_test)

    # Default lookup gives logistic regression
    classifier = linear_model_resolver.make(None)
    classifier.fit(x_train, y_train)
    assert 0.7 < classifier.score(x_test, y_test)

.. seealso:: https://scikit-learn.org/stable/modules/classes.html#linear-classifiers
"""
