import json

import click
import click_config_file
import os
from .database import management as dbmanagement
from .database import orm
import shutil
import pandas as pd
from .filtering.filter import UTRFilter, BuscoFilter, AltSpliceFilter, exemptions_as_sp_names
from .helpers import divvy_up_gpu_indices


@click.group()
@click_config_file.configuration_option(config_file_name='config.ini')
def cli():
    pass


@cli.command()
@click.option('--species-full', required=True)
@click.option('--species-subset', required=True)
@click.option('--working-dir', required=True)
@click.option('--tree', required=True)
@click.option('--nni-config', required=True)
@click.option('--exact-match', is_flag=True)
@click.option('--passed-meta-filter', help='output of spselect metafilter')
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
def setup(working_dir, species_full, species_subset, tree, nni_config, exact_match, passed_meta_filter,
          tuner_gpu_indices):
    """prepares sqlitedb, species weighting and splitting, etc for later use"""
    # check naming in species full/subset that everything matches
    list_full = os.listdir(species_full)
    list_subset = os.listdir(species_subset)
    for full, subset in zip(sorted(list_full), sorted(list_subset)):
        assert full == subset, f"{full} != {subset} when confirming matching species between directories"

    # take all if filtered aren't supplied
    if passed_meta_filter is None:
        quality_sp = list_full.copy()
    else:
        quality_sp = []
        with open(passed_meta_filter) as f:
            for line in f:
                quality_sp.append(line.rstrip())

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
    dbmanagement.add_species_from_tree(tree, list_full, session, exact_match, quality_sp)
    gpu_indices = divvy_up_gpu_indices(tuner_gpu_indices)
    for split in [0, 1]:
        # ID to split because it needs to make two _different_ rounds at the start (should clean)
        r = dbmanagement.RoundHandler(session, split=split, id=split, gpu_indices=gpu_indices[split])
        # randomly select training seed species for each set
        r.set_first_seeds()
        # initialize and prep first round (seed training, adjustment training, model renaming (more symlinks), eval
        r.setup_data()
        r.setup_control_files()
        r.start_seed_training()


@cli.command("next")
@click.option('--working-dir', required=True)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
def ss_next(working_dir, tuner_gpu_indices):
    click.echo(f'will setup and run next step for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    gpu_indices = divvy_up_gpu_indices(tuner_gpu_indices)
    for split in [0, 1]:
        latest_round_id = max(x.id for x in (session.query(orm.Round).filter(orm.Round.split == split).all()))
        r = dbmanagement.RoundHandler(session, split, latest_round_id, gpu_indices=gpu_indices[split])
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
            new_r = dbmanagement.RoundHandler(session, split, latest_round_id + 1, gpu_indices=gpu_indices[split])
            new_r.adjust_seeds_since(r)
            new_r.setup_data()
            new_r.setup_control_files()
            new_r.start_seed_training()


@cli.command()
@click.option('--meta-csv', required=True)
@click.option('--geenuff-csv', required=True)
@click.option('--file-out', required=True)
@click.option('--alt-splice-min', help='minimum alternative transcripts / gene', default=0.001)
@click.option('--utr-min', help='minimum utr/expected utr (expected is mRNA * 2)', default=0.01)
@click.option('--busco-loss-max', help='maximum drop in complete buscos between proteome and genome', default=0.05)
@click.option('--exempt', help='json string with {rule name: [phylo groups, ...], ...} '
                               'that are exempted from filtering. Rule names are "UTR", "busco", "altsplice". '
                               'see filtering/filter/exemptions_as_sp_names for more info.',
              default='{}')
@click.option('--tree')
@click.option('--exact-match', is_flag=True)
def metafilter(meta_csv, geenuff_csv, file_out, alt_splice_min, utr_min, busco_loss_max, exempt, tree, exact_match):
    exempt_dict = json.loads(exempt)
    if exempt_dict:
        assert tree is not None, "if '--exempt' is set, a phylogenetic tree is required under '--tree'"
    meta_dat = pd.read_csv(meta_csv)
    geenuff_dat = pd.read_csv(geenuff_csv)

    remaining = list(meta_dat.loc[:, 'species'])
    exemptions = exemptions_as_sp_names(exempt_dict, remaining, tree_path=tree, exact_match=exact_match)
    as_filt = AltSpliceFilter(dat=geenuff_dat, threshold=alt_splice_min, exempt=exemptions['altsplice'])
    utr_filt = UTRFilter(dat=geenuff_dat, threshold=utr_min, exempt=exemptions['UTR'])
    busco_filt = BuscoFilter(dat=meta_dat, threshold=busco_loss_max, exempt=exemptions['busco'])

    for filt in [as_filt, utr_filt, busco_filt]:
        print(len(remaining))
        remaining = filt.filter(remaining)
    print(len(remaining))
    with open(file_out, 'w') as f:
        f.write('\n'.join(remaining))
