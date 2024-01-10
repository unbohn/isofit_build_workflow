import logging
import os
import sys

import click
from mlky import Config

Config._opts.convertListTypes = False
Config.Null._warn = False


def getSizeGDAL(file):
    """ """
    return 1, 2

    data = gdal.Open(file, gdal.GA_ReadOnly)
    return data.RasterXSize, data.RasterYSize


def apply_oe():
    """ """
    if len(Config.num_neighbors) > 1:
        if Config.empirical_line:
            Logger.error(
                "Empircal Line algorithm cannot be used with greater than 1 num_neighbors"
            )
            return

        if not Config.analytical_line:
            Logger.warning(
                "Analytical Line was not set in the config but should be, enabling now"
            )
            Config.analytical_line = True

    rdn_size = getSizeGDAL(Config.input_radiance)
    for file in (Config.input_loc, Config.input_obs):
        size = getSizeGDAL(file)
        if size != rdn_size:
            Logger.error(
                f"Input file does not match input radiance size, expected {rdn_size} got {size} for file: {file}"
            )
            return

    print("Done")


#%%
@click.group(name="apply_oe")
def cli():
    """\
    apply_oe demo using mlky
    """
    ...


@cli.command(name="run")
@click.option(
    "config",
    help="Configuration YAML",
)
@click.argument("input_radiance")
@click.argument("input_loc")
@click.argument("input_obs")
@click.argument("working_directory")
@click.argument("sensor")
@click.option("-p", "--patch", help="Sections to patch with")
@click.option("-d", "--defs", help="Definitions file", default="mlky_oe.defs.yml")
@click.option("--print", help="Prints the configuration to terminal", is_flag=True)
def main(config, patch, defs, print, input_radiance):
    """\
    Executes the main processes
    """
    # Initialize the global configuration object
    Config(config, patch, defs=defs)

    # Accept CLI override, otherwise default to yaml
    Config.input_radiance = input_radiance or Config.input_radiance
    Config.input_loc = input_loc or Config.input_loc
    Config.input_obs = input_obs or Config.input_obs
    Config.working_directory = working_directory or Config.working_directory
    Config.sensor = sensor or Config.sensor

    if print:
        click.echo(Config.dumpYaml())

    # Logging handlers
    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, Config.log.terminal))
    handlers.append(sh)

    if Config.log.file:
        if Config.log.mode == "write" and os.path.exists(Config.log.file):
            os.remove(Config.log.file)

        # Add the file logging
        fh = logging.FileHandler(Config.log.file)
        fh.setLevel(Config.log.level)
        handlers.append(fh)

    logging.basicConfig(
        level=getattr(logging, Config.log.level),
        format=Config.log.format
        or "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt=Config.log.datefmt or "%m-%d %H:%M",
        handlers=handlers,
    )

    if Config.validate():
        apply_oe()
    else:
        click.echo("Please correct any configuration errors before proceeding")


@cli.command(name="generate")
@click.option(
    "-f", "--file", help="File to write the template to", default="mlky_oe.template.yml"
)
@click.option("-d", "--defs", help="Definitions file", default="mlky_oe.defs.yml")
def generate(file, defs):
    """\
    Generates a default config template using the definitions file
    """
    Config(data={}, defs=defs)
    Config.generateTemplate(file=file)
    click.echo(f"Wrote template configuration to: {file}")


if __name__ == "__main__":
    cli()
