"""Tests for the scikit-learn contribution module."""

import unittest

try:
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None  # pragma: no cover


@unittest.skipUnless(sklearn, "Can not test sklearn contrib without ``pip install scikit-learn``.")
class TestSklearn(unittest.TestCase):
    """Test for the scikit-learn contribution module."""

    def test_classifier_resolver(self) -> None:
        """Tests for the classifier resolver."""
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        from class_resolver.contrib.sklearn import classifier_resolver

        x, y = datasets.load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        classifier = classifier_resolver.make("LogisticRegression")
        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)
        self.assertLessEqual(0.0, accuracy)
        self.assertGreaterEqual(1.0, accuracy)

        for name, cls in classifier_resolver.lookup_dict.items():
            with self.subTest(name=name):
                classifier = cls()
                classifier.fit(x_train, y_train)
                accuracy = classifier.score(x_test, y_test)
                self.assertLessEqual(0.0, accuracy)
                self.assertGreaterEqual(1.0, accuracy)
