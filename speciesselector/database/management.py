import os
import shutil
import subprocess
import time

from . import orm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ete3 import Tree
from speciesselector.helpers import match_tree_names_plants, match_tree_names_exact, parse_eval_log, F1Decode
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


def add_species_from_tree(tree_path, sp_names, session, exact_match, quality_sp):
    newicktree = Tree(tree_path)
    tree_names = [x.name for x in newicktree.get_descendants() if x.is_leaf()]
    if exact_match:
        match_fn = match_tree_names_exact
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


def mkdir_neg_p(path_fn):
    def inner(*args, **kwargs):
        path = path_fn(*args, **kwargs)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    return inner


class PathMaker:
    """convenience path collection for consistency and also just tab completion accessibility"""

    seed_str = 'seed'
    adj_str = 'adjustment'
    test_data = 'test_data.h5'
    data_str = 'data'
    training_str = 'training'
    eval_str = 'evaluation'
    nni_str = 'nni'
    models_str = 'models'
    config_yml = 'nni_config.yml'
    search_space_json = 'search_space.json'

    def __init__(self, session):
        self.session = session
        self.cache = {}

        home = os.environ['HOME']
        self.nni_home = ospj(home, 'nni-experiments')

    def path_by_name(self, name):
        # pull from db only once per PathMaker instance
        if name not in self.cache:
            res = self.session.query(orm.Path).filter(orm.Path.name == name).all()
            assert len(res) == 1, f'got != one path for {name}: {res}'
            self.cache[name] = res[0].value
        return self.cache[name]

    @property
    def working_dir(self):
        return self.path_by_name('working_dir')

    @property
    def species_full(self):
        return self.path_by_name('species_full')

    @property
    def species_subset(self):
        return self.path_by_name('species_subset')

    @property
    def config_template(self):
        return self.path_by_name('config_template')

    @property
    def phylo_tree(self):
        return self.path_by_name('phylo_tree')

    def full_h5(self, species_name):
        return ospj(self.species_full, species_name, self.test_data)

    def subset_h5(self, species_name):
        return ospj(self.species_subset, species_name, self.test_data)

    # round related dynamic
    @staticmethod
    def adj_str_sp(adj_sp):
        return f'adjustment_{adj_sp}'

    def sp_from_adj_str(self, adj_str):
        return re.sub(self.adj_str_sp(''), '', adj_str)  # remove "adjustment_" to leave just species

    # round / data related
    def round(self, rnd):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        return ospj(rtemp.format(rnd.id), stemp.format(rnd.split))

    @property
    @mkdir_neg_p
    def data(self):
        return ospj(self.working_dir, self.data_str)

    @mkdir_neg_p
    def data_round(self, rnd):
        return ospj(self.data, self.round(rnd))

    @mkdir_neg_p
    def data_round_seed(self, rnd):
        return ospj(self.data_round(rnd), self.seed_str)

    @mkdir_neg_p
    def data_round_adj(self, rnd):
        return ospj(self.data_round(rnd), self.adj_str)

    @mkdir_neg_p
    def data_round_adj_sp(self, rnd, species_name):
        return ospj(self.data_round(rnd), self.adj_str_sp(species_name))

    # nni paths
    @property
    @mkdir_neg_p
    def nni(self):
        return ospj(self.working_dir, self.nni_str)

    @property
    @mkdir_neg_p
    def nni_training_seed(self):
        return ospj(self.nni, self.training_str, self.seed_str)

    @property
    @mkdir_neg_p
    def nni_training_adj(self):
        return ospj(self.nni, self.training_str, self.adj_str)

    @property
    @mkdir_neg_p
    def nni_evaluation_adj(self):
        return ospj(self.nni, self.eval_str, self.adj_str)

    @mkdir_neg_p
    def nni_training_seed_round(self, rnd):
        return ospj(self.nni_training_seed, self.round(rnd))

    @mkdir_neg_p
    def nni_training_adj_round(self, rnd):
        return ospj(self.nni_training_adj, self.round(rnd))

    @mkdir_neg_p
    def nni_evaluation_adj_round(self, rnd):
        return ospj(self.nni_evaluation_adj, self.round(rnd))

    # model organization
    @property
    @mkdir_neg_p
    def models(self):
        return ospj(self.working_dir, self.models_str)

    @mkdir_neg_p
    def models_round(self, rnd):
        return ospj(self.models, self.round(rnd))

    @mkdir_neg_p
    def models_round_seed(self, rnd):
        return ospj(self.models_round(rnd), self.seed_str)

    def h5_round_seed(self, rnd, model='best_model.h5'):
        return ospj(self.models_round_seed(rnd), model)

    @mkdir_neg_p
    def models_round_adj(self, rnd):
        return ospj(self.models_round(rnd), self.adj_str)

    def h5_round_adj(self, rnd, model='best_model.h5'):
        return ospj(self.models_round_adj(rnd), model)

    @mkdir_neg_p
    def models_round_adj_sp(self, rnd, species_name):
        return ospj(self.models_round(rnd), self.adj_str_sp(species_name))

    def h5_round_adj_sp(self, rnd, species_name, model='best_model.h5'):
        return ospj(self.models_round_adj_sp(rnd, species_name), model)

    def h5_round_custom(self, rnd, custom, model='best_model.h5'):
        return ospj(self.models_round(rnd), custom, model)


def robust_4_me_symlink(src, dest):
    # overwrite a destination symlink (and only symlink)
    if os.path.exists(dest):
        if os.path.islink(dest):
            os.remove(dest)
    # make sure that what one is linking to in fact exists
    # (for the sake of throwing a timely error if something isn't working)
    assert os.path.exists(src), f"file to link to: {src} not found"
    os.symlink(src, dest)


class RoundHandler:
    def __init__(self, session, split, id, gpu_indices):
        self.session = session
        self.split = split
        self.id = id
        self.gpu_indices = gpu_indices
        self.base_port = 8080  # todo, parameterize to allow user flexible port usage
        self.pm = PathMaker(session)

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

    def set_first_seeds(self, max_n_seeds=8):
        # get all species in set
        set_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).\
            filter(orm.Species.is_quality).all()
        max_n_seeds = min(max_n_seeds, len(set_sp) // 2)  # leave at least half for validation
        # select trainers
        random.shuffle(set_sp)
        trainers = set_sp[:max_n_seeds]
        # setup seed model & seed trainers
        seed_model = orm.SeedModel(split=self.split, round=self.round)
        seed_trainers = [orm.SeedTrainingSpecies(species=s, seed_model=seed_model) for s in trainers]
        self.session.add(seed_model)
        self.session.add_all(seed_trainers)
        self.session.commit()

    @property
    def seed_model(self):
        seed_model = self.session.query(orm.SeedModel).filter(orm.SeedModel.round_id == self.id).\
            filter(orm.SeedModel.split == self.split).all()
        assert len(seed_model) == 1, f"{len(seed_model)} != 1, seed models found?..."
        return seed_model[0]

    @property
    def seed_training_species(self):
        return [ssp.species for ssp in self.seed_model.seed_training_species]

    @property
    def seed_validation_species(self):
        split_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).\
            filter(orm.Species.is_quality).all()
        return [sp for sp in split_sp if sp not in self.seed_training_species]

    def setup_data(self):
        # seed trainers
        seed_sp_t = self.seed_training_species
        # seed validation AND adjustment species that will be added 1 by 1 to trainers
        seed_sp_v = self.seed_validation_species

        # setup seed data
        # seed training prep with 'full' h5 files
        seed_dir = self.pm.data_round_seed(self)
        adj_dir = self.pm.data_round_adj(self)

        def dest(dest_dir, species_name, is_train=True):
            """final path to symlink train/val h5 files to"""
            if is_train:
                h5 = f'training_data.{species_name}.h5'
            else:
                h5 = f'validation_data.{species_name}.h5'
            return ospj(dest_dir, h5)

        for sp in seed_sp_t:
            # for seed train
            robust_4_me_symlink(self.pm.full_h5(sp.name), dest(seed_dir, sp.name))
            # for fine-tuning w/o changing species, but w/ subset etc to match other adjustments
            robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_dir, sp.name))
        for sp in seed_sp_v:
            robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(seed_dir, sp.name, is_train=False))
            robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_dir, sp.name, is_train=False))
        # setup adjusted data
        # one by one addition of validation species to train
        for adj_sp in seed_sp_v:
            adj_sp_dir = self.pm.data_round_adj_sp(self, adj_sp.name)
            for sp in seed_sp_t + [adj_sp]:  # same trainers as seed + adjustment species
                robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name))
            for sp in seed_sp_v:
                if not sp == adj_sp:  # same validators as seed - adjustment species
                    robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name, is_train=False))
        # one by one drop of training species from train
        for adj_sp in seed_sp_t:
            adj_sp_dir = self.pm.data_round_adj_sp(self, adj_sp.name)
            for sp in seed_sp_t:
                if not sp == adj_sp:  # - adjustment species from train
                    robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name))
            for sp in seed_sp_v + [adj_sp]:  # + adjustment species to val
                robust_4_me_symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name, is_train=False))

    def setup_control_files(self):
        # train seed
        configgy = Configgy(self.pm.config_template, self)
        cfg_ssj = configgy.fmt_train_seed()
        configgy.write_to(cfg_ssj, self.pm.nni_training_seed_round(self))
        # train adjustments
        cfg_ssj = configgy.fmt_train_adjust()
        configgy.write_to(cfg_ssj, self.pm.nni_training_adj_round(self))
        # evaluation adjustments
        cfg_ssj = configgy.fmt_evaluations()
        configgy.write_to(cfg_ssj, self.pm.nni_evaluation_adj_round(self))
        self.round.status = "prepped"
        self.session.add(self.round)
        self.session.commit()

    def cp_control_files(self, _from):
        exp_id = self.get_nni_id()
        start_dir = ospj(self.pm.nni_home, exp_id, 'start')
        os.mkdir(start_dir)
        shutil.copy(ospj(_from, 'search_space.json'), start_dir)
        shutil.copy(ospj(_from, 'nni_config.yml'), start_dir)
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
        elif record_to == 'nni_adjustment_id':
            self.round.nni_adjustment_id = nni_exp_id
        elif record_to == 'nni_eval_id':
            self.round.nni_eval_id = nni_exp_id
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
        self.start_nni(status_in="prepped",
                       status_out="seeds_training",
                       nni_dir=self.pm.nni_training_seed_round(self),
                       record_to='nni_seeds_id')

    def check_and_link_seed_results(self):
        assert self.round.status.name == "seeds_training"
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round.nni_seeds_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) == 1, "expected exactly 1 trial for seed training of round {}/split{}".format(
            self.id, self.split)
        trial = trials[0]
        # link best model
        robust_4_me_symlink(ospj(trial_base, trial, 'best_model.h5'), self.pm.h5_round_seed(self))
        self.stop_nni()

    def start_adj_training(self):
        self.start_nni(status_in="seeds_training",
                       status_out="adjustments_training",
                       nni_dir=self.pm.nni_training_adj_round(self),
                       record_to='nni_adjustment_id')

    def check_and_link_adj_results(self):
        assert self.round.status.name == "adjustments_training"
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round.nni_adjustment_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) > 1, "expected multiple adjustment trials but found only {} for round{}/split{}".format(
            len(trials), self.id, self.split)
        # link best model
        to_add = []
        for trial in trials:
            # todo mv path to pm, maybe most of this loop actually
            with open(ospj(trial_base, trial, 'parameter.cfg')) as f:
                pars = json.load(f)
            data_dir = pars['parameters']['data_dir']
            full_adj_str = data_dir.split('/')[-1]
            robust_4_me_symlink(ospj(trial_base, trial, 'best_model.h5'), self.pm.h5_round_custom(self, full_adj_str))
            # also create a db entry for the AdjustmentModel
            sp_id, delta_n_species = self.sp_id_and_delta_from_adj_str(full_adj_str)
            # check if we've already recorded this (e.g. bc we're restarting failed run)
            existing_matches = self.session.query(orm.AdjustmentModel).\
                filter(orm.AdjustmentModel.round_id == self.id).\
                filter(orm.AdjustmentModel.species_id == sp_id).all()
            print(existing_matches, '<- existing matches')
            if not existing_matches:
                adj_model = orm.AdjustmentModel(round=self.round, species_id=sp_id,
                                                delta_n_species=delta_n_species, seed_model=self.seed_model)
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
            if species in self.seed_training_species:
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

    def start_adj_evaluation(self):
        self.start_nni(status_in="adjustments_training",
                       status_out="evaluating",
                       nni_dir=self.pm.nni_evaluation_adj_round(self),
                       record_to='nni_eval_id')

    def check_and_process_evaluation_results(self):
        assert self.round.status.name == "evaluating"
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round.nni_eval_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) > 1, "expected multiple adjustment trials but found only {} for round{}/split{}".format(
            len(trials), self.id, self.split)
        # link best model
        to_add = []
        for trial in trials:
            # todo mv path to pm, maybe most of this loop actually
            with open(ospj(trial_base, trial, 'parameter.cfg')) as f:
                pars = json.load(f)
            test_data = pars['parameters']['test_data']
            test_sp = self.sp_by_name(test_data.split('/')[-2])
            load_model_path = pars['parameters']['load_model_path']
            full_adj_str = load_model_path.split('/')[-2]  # -1 is best_model.h5, -2 is adj species
            # record data/adj combo
            results = parse_eval_log(ospj(trial_base, trial, 'trial.log'))
            sp_id, delta_n_species = self.sp_id_and_delta_from_adj_str(full_adj_str)
            adjustment_model = self.session.query(orm.AdjustmentModel).\
                filter(orm.AdjustmentModel.species_id == sp_id).\
                filter(orm.AdjustmentModel.round_id == self.id).all()
            assert len(adjustment_model) == 1, f"{len(adjustment_model)} adjustment model found for {sp_id} in round {self.id}"
            adjustment_model = adjustment_model[0]
            print(results)
            # check if we've already recorded this (e.g. bc we're restarting failed run
            existing_matches = self.session.query(orm.RawResult).\
                filter(orm.RawResult.adjustment_model_id == adjustment_model.id).\
                filter(orm.RawResult.test_species_id == test_sp.id).all()
            if not existing_matches:
                raw_result = orm.RawResult(adjustment_model=adjustment_model,
                                           test_species=test_sp,
                                           genic_f1=results[F1Decode.GENIC],
                                           intergenic_f1=results[F1Decode.IG],
                                           utr_f1=results[F1Decode.UTR],
                                           cds_f1=results[F1Decode.CDS],
                                           intron_f1=results[F1Decode.INTRON])
                to_add.append(raw_result)
        self.session.add_all(to_add)
        self.session.commit()
        # aggregate eval data for the round
        to_add = []
        for adjustment_model in self.round.adjustment_models:
            raw_results = adjustment_model.raw_results
            print(adjustment_model)
            for f1_str in ["genic_f1", "intergenic_f1", "utr_f1", "cds_f1", "intron_f1"]:
                weighted = self.aggregate_res(f1_str, raw_results)
                adjustment_model.set_attr_by_name(f"weighted_test_{f1_str}", weighted)
            to_add.append(adjustment_model)
        self.session.add_all(to_add)
        self.session.commit()
        self.stop_nni()

    @staticmethod
    def aggregate_res(f1_str, results):
        weights = [res.test_species.phylogenetic_weight for res in results]
        print(weights, '<- weights')
        f1s = [res.get_attr_by_name(f1_str) for res in results]
        print(f1s, '<- f1s')
        # divide by weights and not N so that weighted values can be compared _between_ splits
        weighted_res = sum([f1 * w for f1, w in zip(weights, f1s)]) / sum(weights)
        return weighted_res

    def adjust_seeds_since(self, prev_round):
        prev_models = prev_round.round.adjustment_models
        trainers = [x for x in prev_round.seed_training_species]
        # sort descending genic f1
        sorted_prev = sorted(prev_models, key=lambda x: - x.weighted_test_genic_f1)
        for adj_model in sorted_prev:
            # consider only improvements, so quit once the un-changed model has been found
            if adj_model.delta_n_species == 0:
                break
            elif adj_model.delta_n_species == 1:
                trainers.append(adj_model.species)
            elif adj_model.delta_n_species == -1:
                trainers = [x for x in trainers if x != adj_model.species]
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


class Configgy:
    def __init__(self, template_path: str, round_handler: RoundHandler):
        with open(template_path) as f:
            self.config_template = f.readlines()
        self.config_template = [x.rstrip() for x in self.config_template]
        for i, line in enumerate(self.config_template):
            self.config_template[i] = line.rstrip()
            if re.match('^ +command:', line) or re.match('^trialCommand:', line):  # legacy, and 'current' (2022) nni
                self.config_template[i] = line + ' {}'
        self.rh = round_handler

    def config(self, additions=''):
        most = '\n'.join(self.config_template).format(additions)
        if self.rh.gpu_indices is not None:
            most += '\n  gpuIndices: {}'.format(self.rh.gpu_indices)
        return most

    def fmt_train_seed(self):
        search_space = {'data_dir': {'_type': "choice", "_value": [self.rh.pm.data_round_seed(self.rh)]}}
        config_str = self.config()
        return config_str, search_space

    def fmt_train_adjust(self):
        config_str = self.config('--resume-training --load-model-path {}'.format(
            self.rh.pm.h5_round_seed(self.rh)))
        infold_species = self.rh.session.query(orm.Species).filter(orm.Species.split == self.rh.split).\
            filter(orm.Species.is_quality).all()

        data_dirs = [self.rh.pm.data_round_adj(self.rh)] + [self.rh.pm.data_round_adj_sp(self.rh, sp.name)
                                                            for sp in infold_species]
        search_space = {'data_dir': {'_type': "choice", "_value": data_dirs}}
        return config_str, search_space

    def fmt_evaluations(self):
        config_str = self.config('--eval')
        xfold_species = self.rh.session.query(orm.Species).filter(orm.Species.split != self.rh.split).\
            filter(orm.Species.is_quality).all()
        infold_species = self.rh.session.query(orm.Species).filter(orm.Species.split == self.rh.split).\
            filter(orm.Species.is_quality).all()
        test_datas = [self.rh.pm.subset_h5(sp.name) for sp in xfold_species]
        load_model_paths = [self.rh.pm.h5_round_adj(self.rh)] + [self.rh.pm.h5_round_adj_sp(self.rh, sp.name)
                                                                 for sp in infold_species]
        search_space = {'test_data': {'_type': "choice", "_value": test_datas},
                        'load_model_path': {'_type': 'choice', '_value': load_model_paths}}
        return config_str, search_space

    def write_to(self, config_search_space, outdir):
        cfg, search_space = config_search_space
        with open(ospj(outdir, self.rh.pm.config_yml), 'w') as f:
            f.write(cfg)
        with open(ospj(outdir, self.rh.pm.search_space_json), 'w') as f:
            json.dump(search_space, f, indent=4)






