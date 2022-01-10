import os
from . import orm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ete3 import Tree
from speciesselector.helpers import match_tree_names_plants, match_tree_names_exact
import math
import random


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
        return os.path.join(self.species_full, species_name, self.test_data)

    def subset_h5(self, species_name):
        return os.path.join(self.species_subset, species_name, self.test_data)

    # round related dynamic
    @staticmethod
    def adj_str_sp(adj_sp):
        return f'adjustment_{adj_sp}'

    # round / data related
    @mkdir_neg_p
    def round(self, rnd):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        return os.path.join(self.working_dir, rtemp.format(rnd.id), stemp.format(rnd.split))

    @mkdir_neg_p
    def round_data(self, rnd):
        return os.path.join(self.round(rnd), self.data_str)

    @mkdir_neg_p
    def round_data_seed(self, rnd):
        return os.path.join(self.round_data(rnd), self.seed_str)

    @mkdir_neg_p
    def round_data_adj(self, rnd):
        return os.path.join(self.round_data(rnd), self.adj_str)

    @mkdir_neg_p
    def round_data_adj_sp(self, rnd, species_name):
        return os.path.join(self.round_data(rnd), self.adj_str_sp(species_name))

    # round/training related
    @mkdir_neg_p
    def round_training(self, rnd):
        return os.path.join(self.round(rnd), self.training_str)

    @mkdir_neg_p
    def round_training_seed(self, rnd):
        return os.path.join(self.round_training(rnd), self.seed_str)

    @mkdir_neg_p
    def round_training_adj(self, rnd):
        return os.path.join(self.round_training(rnd), self.adj_str)

    @mkdir_neg_p
    def round_training_seed_nni(self, rnd):
        return os.path.join(self.round_training_seed(rnd), self.nni_str)

    @mkdir_neg_p
    def round_training_seed_models(self, rnd):
        return os.path.join(self.round_training_seed(rnd), self.models_str)

    @mkdir_neg_p
    def round_training_adj_nni(self, rnd):
        return os.path.join(self.round_training_adj(rnd), self.nni_str)

    @mkdir_neg_p
    def round_training_adj_models(self, rnd):
        return os.path.join(self.round_training_adj(rnd), self.models_str)


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
            return os.path.join(dest_dir, h5)

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

    def setup_control_files(self):
        # train seed
        with open(self.pm.config_template) as f:
            config_template_text = f.readlines()




