"""Command-line interface."""
import click

# from .stitching import stitch_images


@click.command()
@click.version_option()
def main() -> None:
    """M2Stitch."""


#    testimages, props = test_image_path
#    """It exits with a status code of zero."""
#    rows = props["row"].to_list()
#    cols = props["col"].to_list()
#    result_df, _ = stitch_images(testimages, rows, cols)


if __name__ == "__main__":
    main(prog_name="m2stitch")  # pragma: no cover
