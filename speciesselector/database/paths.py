import os
import re

from . import orm

ospj = os.path.join


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
    config_yml = 'config.yml'
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

    @staticmethod
    def seed_model_str(seed_model):
        return f'seed_{seed_model.id:03}'

    @staticmethod
    def remix_model_str(model0, model1):
        return f'seed_{model0.id:03}_{model1.id:03}'

    @staticmethod
    def sm_id_from_sm_str(sm_str):
        id_str = re.sub('seed_', '', sm_str)
        return int(id_str)

    # round / data related
    def round(self, rnd):
        rtemp, stemp = 'round_{:03}', 'split_{:02}'
        return ospj(rtemp.format(rnd.id), stemp.format(rnd.split))

    def round_remix(self, rnd0, rnd1):
        round_folder = f'round_{rnd0.id:03d}_{rnd1.id:03d}'
        split_folder = f'split_{rnd0.split:02d}_{rnd1.split:02d}'
        return ospj(round_folder, split_folder)

    @property
    @mkdir_neg_p
    def data(self):
        return ospj(self.working_dir, self.data_str)

    @mkdir_neg_p
    def data_round(self, rnd):
        return ospj(self.data, self.round(rnd))

    @mkdir_neg_p
    def data_remixes(self, rnd0, rnd1, model0, model1):
        # e.g. round_000_001/split_00_01/seed_002_016
        rnd = self.round_remix(rnd0, rnd1)
        remix_folder = self.remix_model_str(model0, model1)
        return ospj(self.data, rnd, remix_folder)

    @mkdir_neg_p
    def data_round_seed(self, rnd, seed_model):
        return ospj(self.data_round(rnd), self.seed_model_str(seed_model))

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
    def nni_evaluation_seed(self):
        return ospj(self.nni, self.eval_str, self.seed_str)

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
    def nni_training_remix_round(self, rnd0, rnd1):
        return ospj(self.nni_training_seed, self.round_remix(rnd0, rnd1))

    @mkdir_neg_p
    def nni_evaluation_seed_round(self, rnd):
        return ospj(self.nni_evaluation_seed, self.round(rnd))

    @mkdir_neg_p
    def nni_evaluation_remix_round(self, rnd0, rnd1):
        # leaving it under seed primarily because that's what I did manually...
        return ospj(self.nni_evaluation_seed, self.round_remix(rnd0, rnd1))

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
    def models_round_seed(self, rnd, seed_model):
        return ospj(self.models_round(rnd), self.seed_model_str(seed_model))

    @mkdir_neg_p
    def models_round_remix(self, rnd0, rnd1, model0, model1):
        return ospj(self.models, self.round_remix(rnd0, rnd1), self.remix_model_str(model0, model1))

    def h5_round_seed(self, rnd, seed_model, model='best_model.h5'):
        return ospj(self.models_round_seed(rnd, seed_model), model)

    def h5_round_remix(self, rnd0, rnd1, model0, model1, model='best_model.h5'):
        return ospj(self.models_round_remix(rnd0, rnd1, model0, model1), model)

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

    @staticmethod
    def h5_dest(dest_dir, species_name, is_train=True):
        """final path to symlink train/val h5 files to"""
        if is_train:
            h5 = f'training_data.{species_name}.h5'
        else:
            h5 = f'validation_data.{species_name}.h5'
        return ospj(dest_dir, h5)


def robust_4_me_symlink(src, dest):
    # overwrite a destination symlink (and only symlink)
    if os.path.exists(dest):
        if os.path.islink(dest):
            os.remove(dest)
    # make sure that what one is linking to in fact exists
    # (for the sake of throwing a timely error if something isn't working)
    assert os.path.exists(src), f"file to link to: {src} not found"
    os.symlink(src, dest)
