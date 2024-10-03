"""Indeed."""

import click

from class_resolver.contrib.sklearn import classifier_resolver


@click.command()
@classifier_resolver.get_option("--classifier-1")
@classifier_resolver.get_option("--classifier-2", required=True)
def main(classifier_1: str, classifier_2: str):
    """Do it."""
    click.echo(f"You picked: {classifier_1}")
    click.echo(f"You picked: {classifier_2}")


if __name__ == "__main__":
    main()
