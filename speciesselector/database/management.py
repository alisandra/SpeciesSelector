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
            print('connecting exisiting database at {}'.format(database_path))
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


class RoundHandler:
    seed_str = 'seed'

    def __init__(self, session, split, id):
        self.session = session
        self.split = split
        self.id = id
        rounds = session.query(orm.Round).filter(orm.Round.id == id).all()
        if len(rounds) == 1:
            self.round = rounds[0]
        elif len(rounds) == 0:
            self.round = orm.Round(id=id, status="initialized")
            session.add(self.round)
            session.commit()
        else:
            raise ValueError(f"{len(rounds)} rounds found with id {id}???")
        # setup basic directory
        wdr = self.dir
        for path in [wdr, os.path.join(wdr, 'training'), os.path.join(wdr, 'evaluation'),
                     os.path.join(wdr, 'data')]:
            if not os.path.exists(path):
                os.makedirs(path)

    @property
    def dir(self):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        working_dir = self.session.query(orm.Path).filter(orm.Path.name == "working_dir").all()[0]
        return os.path.join(working_dir, rtemp.format(self.id), stemp.format(self.split))

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
    def full_dir(self):
        return self.session.query(orm.Path).filter(orm.Path.name == 'species_full').first()

    @property
    def subset_dir(self):
        return self.session.query(orm.Path).filter(orm.Path.name == 'species_subset').first()

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

    @property
    def adj_training_dirs(self):
        return ['adjustment_{}'.format(sp.name) for sp in self.seed_training_species]

    def setup_data(self):
        # setup training dir of seeds with 'full' h5 files
        for dpath in [self.seed_str] + self.adj_training_dirs:
            os.makedirs(dpath)

        # setup seed data
        seed_sp_t = self.seed_training_species
        seed_sp_v = self.seed_validation_species
        full_dir = self.full_dir
        subset_dir = self.subset_dir
        for sp in seed_sp_t:
            os.symlink(
                os.path.join(full_dir, sp.name, 'testing_data.h5'),
                os.path.join(self.dir, 'data', self.seed_str, 'training_data.{}.h5'.format(sp.name)))
        for sp in seed_sp_v:
            os.symlink(
                os.path.join(subset_dir, sp.name, 'testing_data.h5'),
                os.path.join(self.dir, 'data', self.seed_str, 'validation_data.{}.h5'.format(sp.name))
            )
        # setup adjusted data


