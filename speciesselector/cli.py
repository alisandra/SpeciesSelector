import json

import click
import click_config_file
import os
from .database import management as dbmanagement
from .database import orm
from .database.remix import RemixHandler
import shutil
import pandas as pd
from .filtering.filter import UTRFilter, BuscoFilter, AltSpliceFilter, exemptions_as_sp_names
from .helpers import (divvy_up_gpu_indices,
                      split_list, parse_gpu_indices)


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
@click.option('--refseq-match', is_flag=True)
@click.option('--passed-meta-filter', help='output of spselect metafilter')
@click.option('--base-port', type=int, default=8080)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
@click.option('--n-seeds', type=int, default=3, help='number of seed models to train. N random at first, '
                                                     'then 1 optimized + N - 1 random')
@click.option('--max-seed-training-species', default=8, type=int, help='seed training sets will have the lesser of '
                                                                       'this number or half the available species')
@click.option('--only-split', type=int, default=None,
              help='specify 0 or 1 to start just one split (e.g. in case of previous failure)')
def setup(working_dir, species_full, species_subset, tree, nni_config, exact_match, refseq_match, passed_meta_filter,
          tuner_gpu_indices, n_seeds, max_seed_training_species, only_split, base_port):
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
    dbmanagement.add_species_from_tree(tree, list_full, session, exact_match, refseq_match, quality_sp)
    gpu_indices = divvy_up_gpu_indices(tuner_gpu_indices)
    for split in split_list(only_split):  # [0, 1] unless specified
        # ID to split because it needs to make two _different_ rounds at the start (should clean)
        r = dbmanagement.RoundHandler(session, split=split, id=split, gpu_indices=gpu_indices[split], n_seeds=n_seeds,
                                      max_seed_training_species=max_seed_training_species, base_port=base_port)
        # randomly select training seed species for each set
        r.set_random_seeds()
        # initialize and prep first round (seed training, adjustment training, model renaming (more symlinks), eval
        r.setup_seed_data()
        r.setup_seed_control_files()
        r.start_seed_training()


@cli.command("next")
@click.option('--working-dir', required=True)
@click.option('--base-port', type=int, default=8080)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
@click.option('--maximum-changes', type=int, default=6, help='the N largest improvements will be performed '
                                                             '(unless < N improvements are available)')
@click.option('--n-seeds', type=int, default=3, help='number of seed models to train. N random at first, '
                                                     'then 1 optimized + N - 1 random')
@click.option('--max-seed-training-species', default=8, type=int, help='seed training sets will have the lesser of '
                                                                       'this number or half the available species')
@click.option('--only-split', type=int, default=None,
              help='specify 0 or 1 to start just one split (e.g. in case of previous failure)')
def ss_next(working_dir, tuner_gpu_indices, maximum_changes, n_seeds, max_seed_training_species, only_split, base_port):
    click.echo(f'will setup and run next step for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    gpu_indices = divvy_up_gpu_indices(tuner_gpu_indices)
    for split in split_list(only_split):
        latest_round_id = max(x.id for x in (session.query(orm.Round).filter(orm.Round.split == split).all()))
        r = dbmanagement.RoundHandler(session, split, latest_round_id, gpu_indices=gpu_indices[split], n_seeds=n_seeds,
                                      max_seed_training_species=max_seed_training_species, base_port=base_port)
        status = r.round.status.name
        print(f'resuming from status "{status}"')
        # if status was seeds_training, check and record output/nni IDs of above, start fine tuning adjustments
        if status == orm.RoundStatus.seeds_training.name:
            r.check_and_link_seed_results()
            r.start_seed_evaluation()
        elif status == orm.RoundStatus.seeds_evaluating.name:
            r.check_and_process_evaluation_results(is_fine_tuned=False)
            r.setup_adjustment_data()
            r.setup_adj_control_files()
            r.start_adj_training()
        # if status was adjustment training, check and record output/nni IDs of above, start evaluation
        elif status == orm.RoundStatus.adjustments_training.name:
            r.check_and_link_adj_results()
            r.start_adj_evaluation()
        # if status was evaluation, check output of above, record evaluation results, init, prep and start next round
        elif status == orm.RoundStatus.adjustments_evaluating.name:
            r.check_and_process_evaluation_results(is_fine_tuned=True)
            new_r = dbmanagement.RoundHandler(session, split, latest_round_id + 2,  # because two splits
                                              gpu_indices=gpu_indices[split], n_seeds=n_seeds, base_port=base_port,
                                              max_seed_training_species=max_seed_training_species)
            new_r.adjust_seeds_since(r, maximum_changes=maximum_changes)
            new_r.setup_seed_data()
            new_r.setup_seed_control_files()  # end with status 'prepped'
            new_r.start_seed_training()  # end with status 'seeds_training'


@cli.command()
@click.option('--working-dir', required=True)
@click.option('--base-port', type=int, default=8080)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
@click.option('--maximum-changes', type=int, default=6, help='the N largest improvements will be performed '
                                                             '(unless < N improvements are available)')
@click.option('--n-seeds', type=int, default=3, help='number of seed models to train. N random at first, '
                                                     'then 1 optimized + N - 1 random')
@click.option('--max-seed-training-species', default=8, type=int, help='seed training sets will have the lesser of '
                                                                       'this number or half the available species')
@click.option('--only-split', type=int, default=None,
              help='specify 0 or 1 to start just one split (e.g. in case of previous failure)')
def pause(working_dir, tuner_gpu_indices, maximum_changes, n_seeds, max_seed_training_species, only_split, base_port):
    """check and enter results of a run, and prep next without sarting"""
    click.echo(f'will wrap up current step and prep next for {working_dir}')
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    gpu_indices = divvy_up_gpu_indices(tuner_gpu_indices)
    for split in split_list(only_split):
        latest_round_id = max(x.id for x in (session.query(orm.Round).filter(orm.Round.split == split).all()))
        r = dbmanagement.RoundHandler(session, split, latest_round_id, gpu_indices=gpu_indices[split], n_seeds=n_seeds,
                                      max_seed_training_species=max_seed_training_species, base_port=base_port)
        status = r.round.status.name
        print(f'pausing from status "{status}"')
        # if status was seeds_training, check and record output/nni IDs of above
        if status == orm.RoundStatus.seeds_training.name:
            r.check_and_link_seed_results()  # status unchanged
        # if status was seed eval, check and record output, prep adjustment train and eval
        elif status == orm.RoundStatus.seeds_evaluating.name:
            r.check_and_process_evaluation_results(is_fine_tuned=False)
            r.setup_adjustment_data()
            r.setup_adj_control_files()  # end with status ajd prepped
        # if status was adjustment training, check and record output/nni IDs of above
        elif status == orm.RoundStatus.adjustments_training.name:
            r.check_and_link_adj_results()
        # if status was evaluation, check output of above, record evaluation results, init and prep next round
        elif status == orm.RoundStatus.adjustments_evaluating.name:
            r.check_and_process_evaluation_results(is_fine_tuned=True)
            new_r = dbmanagement.RoundHandler(session, split, latest_round_id + 2,  # because two splits
                                              gpu_indices=gpu_indices[split], n_seeds=n_seeds, base_port=base_port,
                                              max_seed_training_species=max_seed_training_species)
            new_r.adjust_seeds_since(r, maximum_changes=maximum_changes)
            new_r.setup_data()
            new_r.setup_control_files()  # end with status seeds prepped


@cli.command()
@click.option('--working-dir', required=True)
@click.option('--base-port', type=int, default=8080)
@click.option('--only-split', type=int, default=None,
              help='specify 0 or 1 to start just one split (e.g. in case of previous failure)')
def resume(working_dir, only_split, base_port):
    """resume (without any result checking or setup) a paused run"""
    click.echo(f'will run next step for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    for split in split_list(only_split):
        latest_round_id = max(x.id for x in (session.query(orm.Round).filter(orm.Round.split == split).all()))
        r = dbmanagement.RoundHandler(session, split, latest_round_id, gpu_indices=None, n_seeds=None,
                                      base_port=base_port, max_seed_training_species=None)
        status = r.round.status.name
        print(f'resuming from status "{status}"')
        # if status was seeds_training, check and record output/nni IDs of above, start fine tuning adjustments
        if status == orm.RoundStatus.seeds_training.name:
            r.start_seed_evaluation()
        elif status == orm.RoundStatus.adjustments_prepped.name:
            r.start_adj_training()
        # if status was adjustment training, start evaluation
        elif status == orm.RoundStatus.adjustments_training.name:
            r.start_adj_evaluation()
        # if status was seeds prepped, start next round
        elif status == orm.RoundStatus.seeds_prepped.name:
            r.start_seed_training()


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


@cli.command()
@click.option('--working-dir', required=True)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
@click.option('--base-port', type=int, default=8080)
def remix_train(working_dir, tuner_gpu_indices, base_port):
    """finds top two models for each split, remixes their species, and starts training"""
    click.echo(f'will setup and run remix for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    gpu_indices = parse_gpu_indices(tuner_gpu_indices)
    latest_round_ids = sorted(x.id for x in (session.query(orm.Round).all()))[-2:]
    r = RemixHandler(session, latest_round_ids, gpu_indices=gpu_indices,
                     base_port=base_port)
    status = r.rounds[0].status.name
    assert status == r.rounds[1].status.name
    print(f'resuming from status "{status}"')
    # store most recent eval results
    if status == orm.RoundStatus.seeds_evaluating.name:
        r.check_and_process_evaluation_results()

    else:
        raise ValueError(f"remix_train only implemented from status {orm.RoundStatus.seeds_evaluating.name}, "
                         f"but current status is {status}")

    r.setup_remix_data()
    r.setup_remix_control_files()
    r.start_remix_training()


@cli.command()
@click.option('--working-dir', required=True)
@click.option('--tuner-gpu-indices', type=str, help='gpu indices to constrain nni to (e.g. 0 or 1-3 or 1,2,3')
@click.option('--base-port', type=int, default=8080)
def remix_eval(working_dir, tuner_gpu_indices, base_port):
    """evaluates remixed models from 'remix_train' on all species"""
    click.echo(f'will eval remix for {working_dir}')
    # if latest round status is 2, start training seeds
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    gpu_indices = parse_gpu_indices(tuner_gpu_indices)
    latest_round_ids = sorted(x.id for x in (session.query(orm.Round).all()))[-2:]
    r = RemixHandler(session, latest_round_ids, gpu_indices=gpu_indices,
                     base_port=base_port)
    status = r.rounds[0].status.name

    if status == orm.RoundStatus.remix_training.name:
        pass
        #r.check_and_link_seed_results()
        #r.start_seed_evaluation()
    else:
        raise ValueError(f"remix_eval only implemented from status {orm.RoundStatus.remix_training.name}, "
                         f"but current status is {status}")


@cli.command()
@click.option('--working-dir', required=True)
def reset_wd(working_dir):
    """changes wd in db to match argument, for moving/testing purposes"""
    engine, session = dbmanagement.mk_session(os.path.join(working_dir, 'spselec.sqlite3'), new_db=False)
    wd_obj = session.query(orm.Path).filter(orm.Path.name == "working_dir").first()
    wd_obj.value = working_dir
    session.add(wd_obj)
    session.commit()
