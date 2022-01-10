import click
import os
from .database import management as dbmanagement
from .database import orm
import shutil


@click.group()
def cli():
    pass


@cli.command()
@click.option('--species-full', required=True)
@click.option('--species-subset', required=True)
@click.option('--working-dir', required=True)
@click.option('--tree', required=True)
@click.option('--nni-config', required=True)
@click.option('--exact-match', is_flag=True)
def setup(working_dir, species_full, species_subset, tree, nni_config, exact_match):
    """prepares sqlitedb, species weighting and splitting, etc for later use"""
    # check naming in species full/subset that everything matches
    list_full = os.listdir(species_full)
    list_subset = os.listdir(species_subset)
    for full, subset in zip(sorted(list_full), sorted(list_subset)):
        assert full == subset, f"{full} != {subset} when confirming matching species between directories"

    # setup db, as well as nni config and search space template
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'))
    shutil.copy(tree, os.path.join(working_dir, 'phylo.tre'))
    shutil.copy(nni_config, os.path.join(working_dir, 'config_template.yml'))  # todo, train and eval versions?

    # what we don't copy, we keep the path to
    new_paths = [orm.Path(name='working_dir', value=working_dir),
                 orm.Path(name='species_full', value=species_full),
                 orm.Path(name='species_subset', value=species_subset)]
    session.add_all(new_paths)
    session.commit()

    # assign weight to species according to depth in tree (deeper, where more species are, is lower)
    # this is so that the trained models are selected for lower phylogenetic bias than the input data
    # this also splits species into sets (2)
    dbmanagement.add_species_from_tree(tree, list_full, session, exact_match)
    # randomly select training seed species for each set
    r = dbmanagement.RoundHandler(session, 0, 0)
    r.set_first_seeds()
    r.setup_data()
    # initialize and prep first round (seed training, adjustment training, model renaming (more symlinks), eval


@cli.command()
@click.option('--working-dir')
def next(working_dir):
    click.echo(f'will setup and run next step for {working_dir}')
    # if latest round status is 2, start training seeds
    # if 3, check and record output/nni IDs of 2, start fine tuning adjustments
    # if 4, check and record output/nni IDs of 3, start evaluation
    # if 5, check output of 4, record evaluation results, init and prep next round
