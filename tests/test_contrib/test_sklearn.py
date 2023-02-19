# -*- coding: utf-8 -*-

"""Tests for the scikit-learn contribution module."""

import unittest

try:
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None  # pragma: no cover


@unittest.skipUnless(sklearn, "Can not test sklearn contrib without ``pip install scikit-learn``.")
class TestSklearn(unittest.TestCase):
    """Test for the scikit-laern contribution module."""

    def test_linear_model(self):
        """Tests for the the linear model resolver."""
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        from class_resolver.contrib.sklearn import linear_model_resolver

        x, y = datasets.load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        classifier = linear_model_resolver.make("LogisticRegression")
        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)
        self.assertLess(0.7, accuracy)
