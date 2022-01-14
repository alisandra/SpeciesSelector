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

    # record paths for later convenience
    new_paths = [orm.Path(name='working_dir', value=working_dir),
                 orm.Path(name='species_full', value=species_full),
                 orm.Path(name='species_subset', value=species_subset),
                 orm.Path(name='config_template', value=os.path.join(working_dir, 'config_template.yml')),
                 orm.Path(name='phylo_tree', value=os.path.join(working_dir, 'phylo.tre'))]
    session.add_all(new_paths)
    session.commit()

    # assign weight to species according to depth in tree (deeper, where more species are, is lower)
    # this is so that the trained models are selected for lower phylogenetic bias than the input data
    # this also splits species into sets (2)
    dbmanagement.add_species_from_tree(tree, list_full, session, exact_match)
    for split in [0, 1]:
        # ID to split because it needs to make two _different_ rounds at the start (should clean)
        r = dbmanagement.RoundHandler(session, split=split, id=split)
        # randomly select training seed species for each set
        r.set_first_seeds()
        # initialize and prep first round (seed training, adjustment training, model renaming (more symlinks), eval
        r.setup_data()
        r.setup_control_files()
        r.start_seed_training()


@cli.command("next")
@click.option('--working-dir')
def ss_next(working_dir):
    click.echo(f'will setup and run next step for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    for split in [0, 1]:
        latest_round_id = max(x.id for x in (session.query(orm.Round).filter(orm.Round.split == split).all()))
        r = dbmanagement.RoundHandler(session, split, latest_round_id)
        status = r.round.status.name
        print(f'resuming from status "{status}"')
        # if status was seeds_training, check and record output/nni IDs of above, start fine tuning adjustments
        if status == "seeds_training":
            r.check_and_link_seed_results()
            r.start_adj_training()
        # if status was adjustment training, check and record output/nni IDs of above, start evaluation
        elif status == "adjustments_training":
            r.check_and_link_adj_results()
            r.start_adj_evaluation()
        # if status was evaluation, check output of above, record evaluation results, init, prep and start next round
        elif status == "evaluating":
            r.check_and_process_evaluation_results()
            new_r = dbmanagement.RoundHandler(session, split, latest_round_id + 1)
            new_r.adjust_seeds_since(r)
            new_r.setup_data()
            new_r.setup_control_files()
            new_r.start_seed_training()
