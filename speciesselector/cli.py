import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--species-full')
@click.option('--species-subset')
@click.option('--working-dir')
@click.option('--tree')
@click.option('--nni-config')
def setup():
    """prepares sqlitedb, species weighting and splitting, etc for later use"""
    click.echo('This will...')
    # check naming in species full/subset as well as tree that everything matches
    # setup db, as well as nni config and search space template
    # assign weight to species according to depth in tree (deeper, where more species are, is lower)
    # this is so that the trained models are selected for lower phylogenetic bias than the input data
    # split species into sets (2)
    # initialize and prep first round


@cli.command()
@click.option('--working-dir')
def next(database):
    click.echo(f'Dropped the database {database}')
    # if latest round status is 2, start training seeds
    # if 3, check output of 2

if __name__ == "__main__":
    cli()
