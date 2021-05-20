"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """M2Stitch."""


if __name__ == "__main__":
    main(prog_name="m2stitch")  # pragma: no cover
