import re
import os
from typing import List
from . import orm
import json

ospj = os.path.join


class Configgy:
    def __init__(self, template_path, round_handler):
        with open(template_path) as f:
            self.config_template = f.readlines()
        self.config_template = [x.rstrip() for x in self.config_template]
        for i, line in enumerate(self.config_template):
            self.config_template[i] = line.rstrip()
            if re.match('^ +command:', line) or re.match('^trialCommand:', line):  # legacy, and 'current' (2022) nni
                self.config_template[i] = line + ' {}'
        self.rh = round_handler

    def config(self, additions=''):
        # todo, replace with a real yml parser/writer
        # this assumes the file ends with a tuner section
        # the indentation will also need to match
        ct = self.config_template.copy()
        if self.rh.gpu_indices is not None:
            i = 0
            old_last_line = ct[-1]
            while old_last_line[i] == " ":
                i += 1
            new_last_line = '{}gpuIndices: {}'.format(' ' * i, self.rh.gpu_indices)
            ct.append(new_last_line)

        most = '\n'.join(ct).format(additions)
        return most

    def fmt_train_seed(self):
        data_dirs = [self.rh.pm.data_round_seed(self.rh, sm) for sm in self.rh.seed_models]
        return self.fmt_train_generic(data_dirs)

    def fmt_train_generic(self, data_dirs):
        search_space = {'data_dir': {'_type': "choice", "_value": data_dirs}}
        config_str = self.config()
        return config_str, search_space

    def fmt_train_adjust(self):
        seed_model = self.rh.best_seed_model()
        config_str = self.config('--resume-training --load-model-path {}'.format(
            self.rh.pm.h5_round_seed(self.rh, seed_model)))
        infold_species = self.rh.session.query(orm.Species).filter(orm.Species.split == self.rh.split).\
            filter(orm.Species.is_quality).all()

        data_dirs = [self.rh.pm.data_round_adj(self.rh)] + [self.rh.pm.data_round_adj_sp(self.rh, sp.name)
                                                            for sp in infold_species]
        search_space = {'data_dir': {'_type': "choice", "_value": data_dirs}}
        return config_str, search_space

    def fmt_adj_evaluations(self):
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

    def fmt_seed_evaluations(self):
        xfold_species = self.rh.session.query(orm.Species).filter(orm.Species.split != self.rh.split).\
            filter(orm.Species.is_quality).all()
        seed_models = self.rh.seed_models
        load_model_paths = [self.rh.pm.h5_round_seed(self.rh, sm) for sm in seed_models]
        return self.fmt_evaluations_generic(xfold_species, load_model_paths)

    def fmt_remix_evaluations(self):
        # in contrast to other evaluations, this will simply be on _all_ species; as one will want this anyway.
        # which to consider in selecting the best will be handled post-hoc
        all_species = self.rh.session.query(orm.Species).filter(orm.Species.is_quality).all()
        remix_models = self.rh.remix_models
        rounds = self.rh.round_handlers
        load_model_paths = [self.rh.pm.h5_round_remix(rounds[0], rounds[1], m0, m1) for m0, m1 in remix_models]
        return self.fmt_evaluations_generic(species=all_species, load_model_paths=load_model_paths)

    def fmt_evaluations_generic(self, species: List[orm.Species], load_model_paths: List[str]):
        config_str = self.config('--eval')
        test_datas = [self.rh.pm.subset_h5(sp.name) for sp in species]
        search_space = {'test_data': {'_type': "choice", "_value": test_datas},
                        'load_model_path': {'_type': 'choice', '_value': load_model_paths}}
        return config_str, search_space

    def write_to(self, config_search_space, outdir):
        cfg, search_space = config_search_space
        with open(ospj(outdir, self.rh.pm.config_yml), 'w') as f:
            f.write(cfg)
        with open(ospj(outdir, self.rh.pm.search_space_json), 'w') as f:
            json.dump(search_space, f, indent=4)
