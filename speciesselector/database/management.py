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

    @staticmethod
    def adj_str(adj_sp):
        return f'adjustment_{adj_sp}'

    @property
    def dir(self):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        working_dir = self.session.query(orm.Path).filter(orm.Path.name == "working_dir").all()[0].value
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
        return self.session.query(orm.Path).filter(orm.Path.name == 'species_full').first().value

    @property
    def subset_dir(self):
        return self.session.query(orm.Path).filter(orm.Path.name == 'species_subset').first().value

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
        full_dir = self.full_dir
        subset_dir = self.subset_dir
        seed_dir = os.path.join(self.dir, 'data', self.seed_str)
        os.makedirs(seed_dir)
        for sp in seed_sp_t:
            os.symlink(
                os.path.join(full_dir, sp.name, 'test_data.h5'),
                os.path.join(seed_dir, f'training_data.{sp.name}.h5'))
        for sp in seed_sp_v:
            os.symlink(
                os.path.join(subset_dir, sp.name, 'test_data.h5'),
                os.path.join(seed_dir, f'validation_data.{sp.name}.h5')
            )
        # setup adjusted data
        for adj_sp in seed_sp_v:
            adj_dir = os.path.join(self.dir, 'data', self.adj_str(adj_sp.name))
            os.makedirs(adj_dir)
            for sp in seed_sp_t + [adj_sp]:  # same trainers as seed + adjustment species
                os.symlink(
                    os.path.join(subset_dir, sp.name, 'test_data.h5'),
                    os.path.join(adj_dir, f'training_data.{sp.name}.h5')
                )
            for sp in seed_sp_v:
                if not sp == adj_sp:  # same validators as seed - adjustment species
                    os.symlink(
                        os.path.join(subset_dir, sp.name, 'test_data.h5'),
                        os.path.join(adj_dir, f'validation_data.{sp.name}.h5')
                    )


