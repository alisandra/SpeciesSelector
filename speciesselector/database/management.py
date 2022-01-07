
import os
from . import orm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ete3 import Tree
from speciesselector.helpers import match_tree_names_plants
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


def add_species_from_tree(tree_path, sp_names, session, engine):
    newicktree = Tree(tree_path)
    tree_names = [x.name for x in newicktree.get_descendants() if x.is_leaf()]
    t2skey = match_tree_names_plants(tree_names, sp_names)
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
