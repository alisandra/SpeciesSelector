import os
import shutil

from . import orm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ete3 import Tree
from speciesselector.helpers import match_tree_names_plants, match_tree_names_exact
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


def add_species_from_tree(tree_path, sp_names, session, exact_match):
    newicktree = Tree(tree_path)
    tree_names = [x.name for x in newicktree.get_descendants() if x.is_leaf()]
    if exact_match:
        match_fn = match_tree_names_exact
    else:
        match_fn = match_tree_names_plants
    t2skey = match_fn(tree_names, sp_names)
    new = []
    for d in newicktree.get_descendants():
        ancestors = d.get_ancestors()
        if d.is_leaf():
            sp_name = t2skey[d.name]
            print(sp_name, len(ancestors))
            weight = 1 / math.log10(len(ancestors) + 1)
            new.append(
                orm.Species(name=sp_name, split=random.choice([0, 1]), phylogenetic_weight=weight)
            )
    session.add_all(new)
    session.commit()


def mkdir_neg_p(path_fn):
    def inner(*arg):
        path = path_fn(*arg)
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
    config_yml = 'config.yml'
    search_space_json = 'search_space.json'

    def __init__(self, session):
        self.session = session
        self.cache = {}

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

    # round / data related
    @mkdir_neg_p
    def round(self, rnd):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        return ospj(self.working_dir, rtemp.format(rnd.id), stemp.format(rnd.split))

    @mkdir_neg_p
    def round_data(self, rnd):
        return ospj(self.round(rnd), self.data_str)

    @mkdir_neg_p
    def round_data_seed(self, rnd):
        return ospj(self.round_data(rnd), self.seed_str)

    @mkdir_neg_p
    def round_data_adj(self, rnd):
        return ospj(self.round_data(rnd), self.adj_str)

    @mkdir_neg_p
    def round_data_adj_sp(self, rnd, species_name):
        return ospj(self.round_data(rnd), self.adj_str_sp(species_name))

    # round/training related
    @mkdir_neg_p
    def round_training(self, rnd):
        return ospj(self.round(rnd), self.training_str)

    @mkdir_neg_p
    def round_training_seed(self, rnd):
        return ospj(self.round_training(rnd), self.seed_str)

    @mkdir_neg_p
    def round_training_adj(self, rnd):
        return ospj(self.round_training(rnd), self.adj_str)

    @mkdir_neg_p
    def round_training_seed_nni(self, rnd):
        return ospj(self.round_training_seed(rnd), self.nni_str)

    @mkdir_neg_p
    def round_training_seed_models(self, rnd):
        return ospj(self.round_training_seed(rnd), self.models_str)

    @mkdir_neg_p
    def round_training_adj_nni(self, rnd):
        return ospj(self.round_training_adj(rnd), self.nni_str)

    @mkdir_neg_p
    def round_training_adj_models(self, rnd):
        return ospj(self.round_training_adj(rnd), self.models_str)


class RoundHandler:
    def __init__(self, session, split, id):
        self.session = session
        self.split = split
        self.id = id

        self.pm = PathMaker(session)

        # check for existing round
        rounds = session.query(orm.Round).filter(orm.Round.id == id).all()
        if len(rounds) == 1:
            self.round = rounds[0]  # use existing if there
        elif len(rounds) == 0:
            self.round = orm.Round(id=id, status="initialized")  # create if there is none
            session.add(self.round)
            session.commit()
        else:
            raise ValueError(f"{len(rounds)} rounds found with id {id}???")

    def set_first_seeds(self, n_seeds=8):
        # get all species in set
        set_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).all()
        # select trainers
        random.shuffle(set_sp)
        trainers = set_sp[:n_seeds]
        # setup seed model & seed trainers
        seed_model = orm.SeedModel(split=self.split, round=self.round)
        seed_trainers = [orm.SeedTrainingSpecies(species=s, seed_model=seed_model) for s in trainers]
        self.session.add(seed_model)
        self.session.add_all(seed_trainers)
        self.session.commit()

    @property
    def seed_model(self):
        seed_model = self.session.query(orm.SeedModel).filter(orm.SeedModel.round_id == self.id).\
            filter(orm.SeedModel.split == self.split).all()[0]
        return seed_model

    @property
    def seed_training_species(self):
        return [ssp.species for ssp in self.seed_model.seed_training_species]

    @property
    def seed_validation_species(self):
        split_sp = self.session.query(orm.Species).filter(orm.Species.split == self.split).all()
        return [sp for sp in split_sp if sp not in self.seed_training_species]

    def setup_data(self):
        # seed trainers
        seed_sp_t = self.seed_training_species
        # seed validation AND adjustment species that will be added 1 by 1 to trainers
        seed_sp_v = self.seed_validation_species

        # setup seed data
        # seed training prep with 'full' h5 files
        seed_dir = self.pm.round_data_seed(self)
        adj_dir = self.pm.round_data_adj(self)

        def dest(dest_dir, species_name, is_train=True):
            """final path to symlink train/val h5 files to"""
            if is_train:
                h5 = f'training_data.{species_name}.h5'
            else:
                h5 = f'validation_data.{species_name}.h5'
            return ospj(dest_dir, h5)

        for sp in seed_sp_t:
            # for seed train
            os.symlink(self.pm.full_h5(sp.name), dest(seed_dir, sp.name))
            # for fine-tuning w/o changing species, but w/ subset etc to match other adjustments
            os.symlink(self.pm.subset_h5(sp.name), dest(adj_dir, sp.name))
        for sp in seed_sp_v:
            os.symlink(self.pm.subset_h5(sp.name), dest(seed_dir, sp.name, is_train=False))
            os.symlink(self.pm.subset_h5(sp.name), dest(adj_dir, sp.name, is_train=False))
        # setup adjusted data
        for adj_sp in seed_sp_v:
            adj_sp_dir = self.pm.round_data_adj_sp(self, adj_sp.name)
            for sp in seed_sp_t + [adj_sp]:  # same trainers as seed + adjustment species
                os.symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name))
            for sp in seed_sp_v:
                if not sp == adj_sp:  # same validators as seed - adjustment species
                    os.symlink(self.pm.subset_h5(sp.name), dest(adj_sp_dir, sp.name, is_train=False))
        # todo one by one drop of training species as well

    def setup_control_files(self):
        # train seed
        configgy = Configgy(self.pm.config_template, self)
        cfg_ssj = configgy.fmt_train_seed()
        configgy.write_to(cfg_ssj, self.pm.round_training_seed_nni(self))
        # train adjustments
        cfg_ssj = configgy.fmt_train_adjust()
        configgy.write_to(cfg_ssj, self.pm.round_training_adj_nni(self))
        # todo! evaluation adjustments


class Configgy:
    def __init__(self, template_path: str, round_handler: RoundHandler):
        with open(template_path) as f:
            self.config_template = f.readlines()
        self.config_template = [x.rstrip() for x in self.config_template]
        for i, line in enumerate(self.config_template):
            self.config_template[i] = line.rstrip()
            if re.match('^ +command:', line):
                self.config_template[i] = line + ' {}'
        self.rh = round_handler

    def config(self, additions=''):
        return '\n'.join(self.config_template).format(additions)

    def fmt_train_seed(self):
        search_space = {'data_dir': {'_type': "choice", "_value": [self.rh.pm.round_data_seed(self.rh)]}}
        config_str = self.config()
        return config_str, search_space

    def fmt_train_adjust(self):
        config_str = self.config('--resume-training --load-model-path {}/seed/best_model.h5'.format(
            self.rh.pm.round_training_seed_models(self.rh)))
        data_dirs = [self.rh.pm.round_data_adj(self.rh)] + [self.rh.pm.round_data_adj_sp(self.rh, sp.name)
                                                            for sp in self.rh.seed_validation_species]
        search_space = {'data_dir': {'_type': "choice", "_value": data_dirs}}
        return config_str, search_space

    def write_to(self, config_search_space, outdir):
        cfg, search_space = config_search_space
        with open(ospj(outdir, self.rh.pm.config_yml), 'w') as f:
            f.write(cfg)
        with open(ospj(outdir, self.rh.pm.search_space_json), 'w') as f:
            json.dump(search_space, f, indent=4)






