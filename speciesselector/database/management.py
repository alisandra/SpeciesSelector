from statistics import harmonic_mean
import os
import shutil
import subprocess
import time

from . import orm
from .config import Configgy
from .paths import PathMaker, robust_4_me_symlink

from sqlalchemy import create_engine, not_
from sqlalchemy.orm import sessionmaker
from ete3 import Tree
from speciesselector.helpers import (match_tree_names_plants,
                                     match_tree_names_exact,
                                     parse_eval_log,
                                     F1Decode,
                                     match_tree_names_refseq)
import math
import random
import json
import re

ospj = os.path.join


def mk_session(database_path, new_db=True):
    if os.path.exists(database_path):
        if new_db:
            print('overwriting existing database at {}'.format(database_path))
            os.remove(database_path)
        else:
            print('connecting existing database at {}'.format(database_path))
    database_path = 'sqlite:///' + database_path
    engine = create_engine(database_path, echo=False)
    orm.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    return engine, session


class Toggle:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        ret = self._value
        self._value = 1 - self._value
        return ret


def add_species_from_tree(tree_path, sp_names, session, exact_match, refseq_match, quality_sp):
    newicktree = Tree(tree_path)
    tree_names = [x.name for x in newicktree.get_descendants() if x.is_leaf()]
    if exact_match:
        match_fn = match_tree_names_exact
    elif refseq_match:
        match_fn = match_tree_names_refseq
    else:
        match_fn = match_tree_names_plants
    t2skey = match_fn(tree_names, sp_names)
    new = []
    toggle = Toggle()
    splits = [toggle.value for _ in range(len(tree_names))]  # zeros and ones
    random.shuffle(splits)
    i = 0
    for d in newicktree.get_leaves():
        ancestors = d.get_ancestors()
        sp_name = t2skey[d.name]
        weight = 1 / math.log10(len(ancestors) + 1)
        new.append(
            orm.Species(name=sp_name, split=splits[i], phylogenetic_weight=weight,
                        is_quality=sp_name in quality_sp)
        )
        i += 1
    session.add_all(new)
    session.commit()




class RoundHandler:
    def __init__(self, session, split, id, gpu_indices, n_seeds, max_seed_training_species, base_port):
        self.session = session
        self.split = split
        self.id = id
        self.gpu_indices = gpu_indices
        self.base_port = base_port
        self.pm = PathMaker(session)
        self.n_seeds = n_seeds
        self.max_seed_training_species = max_seed_training_species

        # check for existing round
        rounds = session.query(orm.Round).filter(orm.Round.id == id).all()
        if len(rounds) == 1:
            self.round = rounds[0]  # use existing if there
        elif len(rounds) == 0:
            self.round = orm.Round(id=id, status="initialized", split=split)  # create if there is none
            session.add(self.round)
            session.commit()
        else:
            raise ValueError(f"{len(rounds)} rounds found with id {id}???")

    def set_random_seeds(self, n_already_set=0):
        # get all species in set
        set_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).\
            filter(orm.Species.is_quality).all()
        n_seed_training_sp = min(self.max_seed_training_species,
                                 len(set_sp) // 2)  # leave at least half for validation
        # select trainers
        for _ in range(self.n_seeds - n_already_set):
            random.shuffle(set_sp)
            trainers = set_sp[:n_seed_training_sp]
            # setup seed models & seed trainers
            seed_model = orm.SeedModel(split=self.split, round=self.round)
            seed_trainers = [orm.SeedTrainingSpecies(species=s, seed_model=seed_model) for s in trainers]
            self.session.add(seed_model)
            self.session.add_all(seed_trainers)
        self.session.commit()

    @property
    def seed_models(self):
        seed_models = self.session.query(orm.SeedModel).filter(orm.SeedModel.round_id == self.id).\
            filter(orm.SeedModel.split == self.split).order_by(orm.SeedModel.id).all()
        return seed_models

    @staticmethod
    def seed_model_training_species(seed_model):
        return [ssp.species for ssp in seed_model.seed_training_species]

    def seed_model_validation_species(self, seed_model):
        split_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).\
            filter(orm.Species.is_quality).all()
        seed_training_species = self.seed_model_training_species(seed_model)
        return [sp for sp in split_sp if sp not in seed_training_species]

    def best_seed_models(self, n):
        seed_eval_models = [em for em in self.round.evaluation_models if not em.is_fine_tuned]
        # confirm these have been evaluated
        assert seed_eval_models[0].weighted_test_genic_f1 is not None

        # we will choose the best harmonic mean of genic_f1 and phase_0
        def rate_model(x):
            score = harmonic_mean([x.weighted_test_genic_f1, x.weighted_test_phase_0_f1])
            return score
        # sort descending genic f1 / phase
        seed_eval_models = sorted(seed_eval_models, key=rate_model, reverse=True)
        return [x.seed_model for x in seed_eval_models[:n]]  # seed models that have the best x-fold evaluation

    def best_seed_model(self):
        return self.best_seed_models(1)[0]

    def setup_adjustment_data(self):
        """setup adjustment data for fine-tuning the adjustment model that performed the best"""
        seed_model = self.best_seed_model()
        adj_dir = self.pm.data_round_adj(self)

        # seed trainers
        seed_sp_t = self.seed_model_training_species(seed_model)
        # seed validation AND adjustment species that will be added 1 by 1 to trainers
        seed_sp_v = self.seed_model_validation_species(seed_model)

        for sp in seed_sp_t:
            # for fine-tuning w/o changing species, but otherwise to match other adjustments
            robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(adj_dir, sp.name))
        for sp in seed_sp_v:
            robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(adj_dir, sp.name, is_train=False))

        # setup adjusted data
        # one by one addition of validation species to train
        for adj_sp in seed_sp_v:
            adj_sp_dir = self.pm.data_round_adj_sp(self, adj_sp.name)
            for sp in seed_sp_t + [adj_sp]:  # same trainers as seed + adjustment species
                robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(adj_sp_dir, sp.name))
            for sp in seed_sp_v:
                if not sp == adj_sp:  # same validators as seed - adjustment species
                    robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(adj_sp_dir, sp.name, is_train=False))
        # one by one drop of training species from train
        for adj_sp in seed_sp_t:
            adj_sp_dir = self.pm.data_round_adj_sp(self, adj_sp.name)
            for sp in seed_sp_t:
                if not sp == adj_sp:  # - adjustment species from train
                    robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(adj_sp_dir, sp.name))
            for sp in seed_sp_v + [adj_sp]:  # + adjustment species to val
                robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(adj_sp_dir, sp.name, is_train=False))

    def setup_seed_data(self):
        """setup the seed training data"""
        for seed_model in self.seed_models:
            # seed trainers
            seed_sp_t = self.seed_model_training_species(seed_model)
            # seed validation AND adjustment species that will be added 1 by 1 to trainers
            seed_sp_v = self.seed_model_validation_species(seed_model)

            # setup seed data
            # seed training prep with 'full' h5 files
            seed_dir = self.pm.data_round_seed(self, seed_model)

            for sp in seed_sp_t:
                # for seed train
                robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(seed_dir, sp.name))

            for sp in seed_sp_v:
                robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(seed_dir, sp.name, is_train=False))

    def setup_seed_control_files(self):
        configgy = Configgy(self.pm.config_template, self)
        # train seed
        cfg_ssj = configgy.fmt_train_seed()
        configgy.write_to(cfg_ssj, self.pm.nni_training_seed_round(self))
        # eval seed
        cfg_ssj = configgy.fmt_seed_evaluations()
        configgy.write_to(cfg_ssj, self.pm.nni_evaluation_seed_round(self))
        self.round.status = orm.RoundStatus.seeds_prepped.name
        self.session.add(self.round)
        self.session.commit()

    def setup_adj_control_files(self):
        configgy = Configgy(self.pm.config_template, self)
        # train adjustments
        cfg_ssj = configgy.fmt_train_adjust()  # calls to rh.best_seed_model, i.e. requires train eval to be complete
        configgy.write_to(cfg_ssj, self.pm.nni_training_adj_round(self))
        # evaluation adjustments
        cfg_ssj = configgy.fmt_adj_evaluations()
        configgy.write_to(cfg_ssj, self.pm.nni_evaluation_adj_round(self))
        self.round.status = orm.RoundStatus.adjustments_prepped.name
        self.session.add(self.round)
        self.session.commit()

    def cp_control_files(self, _from):
        exp_id = self.get_nni_id()
        start_dir = ospj(self.pm.nni_home, exp_id, 'start')
        os.mkdir(start_dir)
        shutil.copy(ospj(_from, 'search_space.json'), start_dir)
        shutil.copy(ospj(_from, self.pm.config_yml), start_dir)
        return exp_id

    @property
    def port(self):
        """port for nni, so that splits can run simultaneously and later have results assigned correctly"""
        return self.base_port + self.split

    def get_nni_id(self):
        experiment_list = subprocess.check_output(['nnictl', 'experiment', 'list'])
        # output looks something like the following
        # ----------------------------------------------------------------------------------------
        #                 Experiment information
        # Id: LKx2sAhe    Name: spselec    Status: TUNER_NO_MORE_TRIAL    Port: 8080    Platform: local    StartTime: 2022-01-14 10:15:14    EndTime: N/A
        # Id: loi0MuG2    Name: spselec    Status: TUNER_NO_MORE_TRIAL    Port: 8081    Platform: local    StartTime: 2022-01-14 10:21:24    EndTime: N/A
        #
        # ----------------------------------------------------------------------------------------
        # which experiment is correct can be determined by the port
        experiment_list = experiment_list.decode('utf8')
        port = str(self.port)
        for line in experiment_list.split('\n'):
            target = f'.*Port:\W*{port}'
            if re.match(target, line):
                res = re.sub('Id:\\W*', '', line)  # truncate from start
                res = re.sub('\\W*Name.*', '', res)  # truncate everything after nni id
                return res
        print('Could not parse experiment ID from:\n')
        print(experiment_list)
        raise ValueError()

    def start_nni(self, status_in, status_out, nni_dir, record_to):
        assert self.round.status.name == status_in, "status mismatch: {} != {}".format(self.round.status, status_in)
        sp_args = ['nnictl', 'create', '-c', self.pm.config_yml, '-p', str(self.port)]
        print('passing to subprocess: {}\nwith cwd: {}'.format(sp_args, nni_dir))
        subprocess.run(sp_args, cwd=nni_dir)
        # copy control files to nni directory (as usual/for convenience)
        nni_exp_id = self.cp_control_files(_from=nni_dir)
        # I'm sure there is a better way, but for now
        if record_to == 'nni_seeds_id':
            self.round.nni_seeds_id = nni_exp_id
        elif record_to == "nni_seeds_eval_id":
            self.round.nni_seeds_eval_id = nni_exp_id
        elif record_to == 'nni_adjustments_id':
            self.round.nni_adjustments_id = nni_exp_id
        elif record_to == 'nni_adjustments_eval_id':
            self.round.nni_adjustments_eval_id = nni_exp_id
        else:
            raise ValueError(f'unexpected/unhandled string value: {record_to} for "record_to"')
        # record nni_id in db
        self.round.status = status_out
        self.session.add(self.round)
        self.session.commit()

    def stop_nni(self):
        # stop nni, so that next step can be started
        subprocess.run(['nnictl', 'stop', self.get_nni_id()])
        time.sleep(5)  # temp patch, bc idk how to get `wait` from here...

    def start_seed_training(self):
        self.start_nni(status_in=orm.RoundStatus.seeds_prepped.name,
                       status_out=orm.RoundStatus.seeds_training.name,
                       nni_dir=self.pm.nni_training_seed_round(self),
                       record_to='nni_seeds_id')

    def check_and_link_seed_results(self):
        assert self.round.status.name == orm.RoundStatus.seeds_training.name
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round.nni_seeds_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) == self.n_seeds, "expected exactly {} trials for seed training of round {}/split{}".format(
            self.n_seeds, self.id, self.split)
        to_add = []
        for trial in trials:
            data_name = self.data_dir_frm_json(trial_base, trial)
            seed_model = self.seed_model_from_sm_str(data_name)
            # link all models for loading and eval
            robust_4_me_symlink(ospj(trial_base, trial, 'best_model.h5'), self.pm.h5_round_seed(self, seed_model))
            # and add to db
            existing_matches = self.session.query(orm.EvaluationModel).\
                filter(orm.EvaluationModel.round_id == self.id).\
                filter(orm.EvaluationModel.seed_model_id == seed_model.id).\
                filter(orm.EvaluationModel.is_fine_tuned).all()
            if not existing_matches:
                adj_model = orm.EvaluationModel(round=self.round, is_fine_tuned=False,
                                                delta_n_species=0,  # seeds have no modification
                                                seed_model=seed_model)
                to_add.append(adj_model)
        self.session.add_all(to_add)
        self.session.commit()
        self.stop_nni()

    def start_seed_evaluation(self):
        self.start_nni(status_in=orm.RoundStatus.seeds_training.name,
                       status_out=orm.RoundStatus.seeds_evaluating.name,
                       nni_dir=self.pm.nni_evaluation_seed_round(self),
                       record_to='nni_seeds_eval_id')

    def start_adj_training(self):
        self.start_nni(status_in=orm.RoundStatus.adjustments_prepped.name,
                       status_out=orm.RoundStatus.adjustments_training.name,
                       nni_dir=self.pm.nni_training_adj_round(self),
                       record_to='nni_adjustments_id')

    @staticmethod
    def data_dir_frm_json(trial_base, trial):
        with open(ospj(trial_base, trial, 'parameter.cfg')) as f:
            pars = json.load(f)
        data_dir = pars['parameters']['data_dir']
        if data_dir.endswith('/'):
            data_dir = data_dir[:-1]
        return data_dir.split('/')[-1]

    def check_and_link_adj_results(self):
        assert self.round.status.name == orm.RoundStatus.adjustments_training.name
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round.nni_adjustments_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) > 1, "expected multiple adjustment trials but found only {} for round{}/split{}".format(
            len(trials), self.id, self.split)
        # link best model
        to_add = []
        for trial in trials:
            full_adj_str = self.data_dir_frm_json(trial_base, trial)
            robust_4_me_symlink(ospj(trial_base, trial, 'best_model.h5'), self.pm.h5_round_custom(self, full_adj_str))
            # also create a db entry for the AdjustmentModel
            sp_id, delta_n_species = self.sp_id_and_delta_from_adj_str(full_adj_str)
            # check if we've already recorded this (e.g. bc we're restarting failed run)
            existing_matches = self.session.query(orm.EvaluationModel).\
                filter(orm.EvaluationModel.round_id == self.id).\
                filter(orm.EvaluationModel.species_id == sp_id).\
                filter(orm.EvaluationModel.is_fine_tuned).all()
            if not existing_matches:
                adj_model = orm.EvaluationModel(round=self.round, species_id=sp_id, is_fine_tuned=True,
                                                delta_n_species=delta_n_species, seed_model=self.best_seed_model())
                to_add.append(adj_model)
        self.session.add_all(to_add)
        self.session.commit()
        self.stop_nni()

    def sp_id_and_delta_from_adj_str(self, adj_str):
        """identifies adjustment species (id) and what sort of adjustment was made from filename/adj_str_sp"""
        if adj_str == self.pm.adj_str:
            species_id = None
            delta_n_species = 0
        else:
            sp_name = self.pm.sp_from_adj_str(adj_str)
            species = self.sp_by_name(sp_name)
            species_id = species.id
            if species in self.seed_model_training_species(self.best_seed_model()):
                delta_n_species = -1
            else:
                delta_n_species = 1
        return species_id, delta_n_species

    def sp_by_name(self, sp_name):
        sp = self.session.query(orm.Species).filter(orm.Species.name == sp_name).all()
        if len(sp) == 1:
            return sp[0]
        else:
            raise ValueError(f"Number of species: {sp} found matching '{sp_name}' is not 1!")

    def seed_model_from_sm_str(self, sm_str):
        sm_id = self.pm.sm_id_from_sm_str(sm_str)
        seed_model = self.session.query(orm.SeedModel).filter(orm.SeedModel.id == sm_id).first()
        return seed_model

    def start_adj_evaluation(self):
        self.start_nni(status_in=orm.RoundStatus.adjustments_training.name,
                       status_out=orm.RoundStatus.adjustments_evaluating.name,
                       nni_dir=self.pm.nni_evaluation_adj_round(self),
                       record_to='nni_adjustments_eval_id')

    def check_and_process_evaluation_results(self, is_fine_tuned):
        """from evaluation nni log files to db record there of"""
        if is_fine_tuned:
            assert self.round.status.name == orm.RoundStatus.adjustments_evaluating.name
            trial_base = ospj(self.pm.nni_home, self.round.nni_adjustments_eval_id, 'trials')
        else:
            assert self.round.status.name == orm.RoundStatus.seeds_evaluating.name
            trial_base = ospj(self.pm.nni_home, self.round.nni_seeds_eval_id, 'trials')
        # get trial dir
        trials = os.listdir(trial_base)
        assert len(trials) > 1, "expected multiple adjustment trials but found only {} for round{}/split{}".format(
            len(trials), self.id, self.split)
        # link best model
        for trial in trials:
            results = parse_eval_log(ospj(trial_base, trial, 'trial.log'))
            test_sp, full_train_data_str = self.train_info_from_json(trial_base, trial)
            # record data/adj combo
            if is_fine_tuned:
                # i.e. this is an adjustment model, named by species
                sp_id, delta_n_species = self.sp_id_and_delta_from_adj_str(full_train_data_str)
                eval_model = self.session.query(orm.EvaluationModel).\
                    filter(orm.EvaluationModel.species_id == sp_id).\
                    filter(orm.EvaluationModel.round_id == self.id).\
                    filter(orm.EvaluationModel.is_fine_tuned).all()
                assert len(eval_model) == 1, f"{len(eval_model)} (!= 1) adjustment models found for species: {sp_id} in round {self.id}"
                eval_model = eval_model[0]
            else:
                # i.e. this is a seed model, named by seed_model id
                seed_model = self.seed_model_from_sm_str(full_train_data_str)
                eval_model = self.session.query(orm.EvaluationModel).\
                    filter(orm.EvaluationModel.seed_model_id == seed_model.id).\
                    filter(orm.EvaluationModel.round_id == self.id).\
                    filter(not_(orm.EvaluationModel.is_fine_tuned)).all()
                assert len(eval_model) == 1, f"{len(eval_model)} (!= 1) eval models found for seed model: {seed_model.id} in round {self.id}"
                eval_model = eval_model[0]
            self.add_result2db(results, test_sp, eval_model)
        self.session.commit()
        # aggregate eval data for the round
        to_add = []
        for eval_model in self.round.evaluation_models:
            # skip seeds (is_fine_tuned = False) when running for adjustments
            if eval_model.is_fine_tuned == is_fine_tuned:
                raw_results = eval_model.raw_results
                for f1_str in ["genic_f1", "intergenic_f1", "utr_f1", "cds_f1", "intron_f1", "no_phase_f1",
                               "phase_0_f1", "phase_1_f1", "phase_2_f1"]:
                    weighted = self.aggregate_res(f1_str, raw_results)
                    eval_model.set_attr_by_name(f"weighted_test_{f1_str}", weighted)
                to_add.append(eval_model)
        self.session.add_all(to_add)
        self.session.commit()
        self.stop_nni()

    def train_info_from_json(self, trial_base, trial):
        with open(ospj(trial_base, trial, 'parameter.cfg')) as f:
            pars = json.load(f)
        test_data = pars['parameters']['test_data']
        test_sp = self.sp_by_name(test_data.split('/')[-2])
        load_model_path = pars['parameters']['load_model_path']
        train_dir = load_model_path.split('/')[-2]  # -1 is best_model.h5, -2 is adj species
        return test_sp, train_dir

    def add_result2db(self, results, test_sp, evaluation_model):
        # check if we've already recorded this (e.g. bc we're restarting failed run
        existing_matches = self.session.query(orm.RawResult). \
            filter(orm.RawResult.evaluation_model_id == evaluation_model.id). \
            filter(orm.RawResult.test_species_id == test_sp.id).all()
        if not existing_matches:
            raw_result = orm.RawResult(evaluation_model=evaluation_model,
                                       test_species=test_sp,
                                       genic_f1=results[F1Decode.GENIC],
                                       intergenic_f1=results[F1Decode.IG],
                                       utr_f1=results[F1Decode.UTR],
                                       cds_f1=results[F1Decode.CDS],
                                       intron_f1=results[F1Decode.INTRON],
                                       no_phase_f1=results[F1Decode.NO_PHASE],
                                       phase_0_f1=results[F1Decode.PHASE_0],
                                       phase_1_f1=results[F1Decode.PHASE_1],
                                       phase_2_f1=results[F1Decode.PHASE_2])
            self.session.add(raw_result)

    @staticmethod
    def aggregate_res(f1_str, results):
        weights = [res.test_species.phylogenetic_weight for res in results]
        print(weights, '<- weights')
        f1s = [res.get_attr_by_name(f1_str) for res in results]
        print(f1s, '<- f1s')
        # divide by weights and not N so that weighted values can be compared _between_ splits
        weighted_res = sum([f1 * w for f1, w in zip(weights, f1s)]) / sum(weights)
        return weighted_res

    def adjust_seeds_since(self, prev_round, maximum_changes):
        assert isinstance(prev_round, type(self))
        # take the best seed model, and apply adjustments that were beneficial as fine-tuning
        prev_best_seed_model = prev_round.best_seed_model()
        prev_models = [em for em in prev_round.round.evaluation_models if em.is_fine_tuned]
        trainers = [x for x in prev_round.seed_model_training_species(prev_best_seed_model)]
        # sort descending genic f1
        sorted_prev = sorted(prev_models, key=lambda x: - x.weighted_test_genic_f1)
        n_changed = 0
        for adj_model in sorted_prev:
            # consider only improvements, so quit once the un-changed model has been found
            if adj_model.delta_n_species == 0:
                break
            # also quit if we've already reached maximum changes
            elif n_changed >= maximum_changes:
                break
            elif adj_model.delta_n_species == 1:
                trainers.append(adj_model.species)
            elif adj_model.delta_n_species == -1:
                trainers = [x for x in trainers if x != adj_model.species]
            n_changed += 1
        # add seed model & seed trainers to db
        seed_model = orm.SeedModel(split=self.split, round=self.round)
        seed_trainers = [orm.SeedTrainingSpecies(species=s, seed_model=seed_model) for s in trainers]
        in_split = self.session.query(orm.Species).filter(orm.Species.split == self.split).\
            filter(orm.Species.is_quality).all()
        if len(seed_trainers) >= len(in_split) - 1:
            print('WARNING: less than 2 validation species remaining. Training of Seed and /or adjustment will fail'
                  'due to lack of validation data. This round cannot be completed.')
        self.session.add(seed_model)
        self.session.add_all(seed_trainers)
        self.session.commit()
        # all other seed models are set randomly
        self.set_random_seeds(n_already_set=1)









