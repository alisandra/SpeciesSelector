import itertools
import os
import subprocess

from . import orm
from .paths import PathMaker, robust_4_me_symlink
from .management import RoundHandler, ospj
from .config import Configgy

from sqlalchemy import not_


class NoEntryException(Exception):
    pass


class RemixHandler:
    def __init__(self, session, ids, gpu_indices, base_port):
        self.session = session
        self.ids = ids
        self.gpu_indices = gpu_indices
        self.base_port = base_port
        self.pm = PathMaker(session)

        # assumes two rounds can be found
        rounds = session.query(orm.Round).filter(orm.Round.id.in_(ids)).all()
        assert len(rounds) == 2
        self.rounds = rounds
        self.round_handlers = [RoundHandler(session, split=i % 2, id=i, gpu_indices=None,
                                            base_port=base_port, max_seed_training_species=None, n_seeds=None)
                               for i in ids]

    @property
    def remix_models(self):
        # remix is basically (b0, r0) x (b1, r1), where b=best, r=runner up, and integers refer to the split
        # directories will be b0,b1; b0,r1; r0,b1; r0,r1; in that order
        # remixes will tuple with (round_handler_idx, best_models_idx,
        remixes = []
        best_models_by_round = [r.best_seed_models(2) for r in self.round_handlers]
        for idx0, idx1 in itertools.product([0, 1], [0, 1]):
            model_0 = best_models_by_round[0][idx0]
            model_1 = best_models_by_round[1][idx1]
            remixes.append((model_0, model_1))
        return remixes

    @property
    def remix_dirs(self):
        remix_dirs = [self.pm.data_remixes(self.round_handlers[0],
                                           self.round_handlers[1],
                                           model_0, model_1) for model_0, model_1 in self.remix_models]
        return remix_dirs

    def check_and_process_evaluation_results(self):
        for r in self.round_handlers:
            r.check_and_process_evaluation_results(is_fine_tuned=False)

    def check_and_link_remix_results(self):
        # get trial dir
        rh0 = self.round_handlers[0]
        assert rh0.status.name == orm.RoundStatus.remix_training.name
        trial_base = ospj(self.pm.nni_home, rh0.round.nni_remix_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) == len(self.remix_models)
        to_add = []
        for trial in trials:
            data_name = rh0.data_dir_frm_json(trial_base, trial)  # e.g. seed_020_036
            seed_models = self.seed_models_from_sm_str(data_name)
            # link all models for loading and eval
            robust_4_me_symlink(ospj(trial_base, trial, 'best_model.h5'),
                                self.pm.h5_round_remix(*self.round_handlers + seed_models))
            # and add to the db
            remix_seed_model = self.get_or_make_remix_model_entry(seed_models)
            existing_matches = self.session.query(orm.EvaluationModel).\
                filter(orm.EvaluationModel.round_id == rh0.id).\
                filter(orm.EvaluationModel.seed_model_id == remix_seed_model.id).\
                filter(not_(orm.EvaluationModel.is_fine_tuned)).all()
            if not existing_matches:
                adj_model = orm.EvaluationModel(round=rh0.round, is_fine_tuned=False,
                                                delta_n_species=0,  # seeds have no modification
                                                seed_model=remix_seed_model)
                to_add.append(adj_model)
        self.session.add_all(to_add)
        self.session.commit()
        rh0.stop_nni()

    def training_species(self, models):
        """union of Species objects for two models"""
        ret = self.round_handlers[0].seed_model_training_species(models[0]) + \
            self.round_handlers[1].seed_model_training_species(models[1])
        return ret

    def validation_species(self, training_species):
        """selects all quality non-training species"""
        all_species = self.session.query(orm.Species).\
            filter(orm.Species.is_quality).all()
        return [x for x in all_species if x not in training_species]

    def setup_remix_data(self):
        """setup remix data combining top trainers from both splits to make final candidate models"""
        for models, remix_dir in zip(self.remix_models, self.remix_dirs):
            remix_sp_t = self.training_species(models)
            remix_sp_v = self.validation_species(remix_sp_t)
            for sp in remix_sp_t:
                # for fine-tuning w/o changing species, but otherwise to match other adjustments
                robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(remix_dir, sp.name))

            for sp in remix_sp_v:
                robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(remix_dir, sp.name, is_train=False))

    def make_remix_model_entry(self, models):
        trainers = self.training_species(models)
        seed_model = orm.SeedModel(split=0, round=self.round_handlers[0].round, is_remix=True)
        seed_trainers = [orm.SeedTrainingSpecies(species=s, seed_model=seed_model) for s in trainers]
        self.session.add(seed_model)
        self.session.add_all(seed_trainers)
        self.session.commit()
        return seed_model

    def make_remix_model_entries(self):
        for models in self.remix_models:
            self.make_remix_model_entry(models)

    def get_or_make_remix_model_entry(self, models):
        try:
            return self.get_remix_model_entry(models)
        except NoEntryException:
            return self.make_remix_model_entry(models)

    def get_remix_model_entry(self, models):
        trainers = self.training_species(models)
        t_set = set([x.name for x in trainers])
        maybe_s_models = self.session.query(orm.SeedModel).filter(orm.SeedModel.is_remix).\
            filter(orm.SeedModel.round_id == self.round_handlers[0].round.id)
        remaining_sm = []
        # because there was no actual saving of the remix seed_model ID into any path / anything that can
        # be regenerated later; we'll figure out which one is which based on species matching the combined
        # species of the two sub models that _were_ saved / used in naming
        # here's hoping this really is that last thing built on this...
        for sm in maybe_s_models:
            sm_trainers = set([x.species.name for x in sm.seed_training_species])
            if t_set == sm_trainers:
                remaining_sm.append(sm)
        if not len(remaining_sm):
            raise NoEntryException
        assert len(remaining_sm) == 1, f'> ({len(remaining_sm)})1 potential seed model found! {remaining_sm}'
        return remaining_sm[0]

    def setup_remix_control_files(self):
        configgy = Configgy(self.pm.config_template, self)
        # train seed
        cfg_ssj = configgy.fmt_train_generic(self.remix_dirs)
        configgy.write_to(cfg_ssj, self.pm.nni_training_remix_round(*self.round_handlers))
        # eval seed
        cfg_ssj = configgy.fmt_remix_evaluations()
        configgy.write_to(cfg_ssj, self.pm.nni_evaluation_remix_round(*self.round_handlers))
        for rh in self.round_handlers:
            rh.round.status = orm.RoundStatus.remix_prepped.name
            self.session.add(rh.round)
        self.session.commit()

    def seed_models_from_sm_str(self, sm_str):
        """from seed models string, to seed_model orm objects"""
        sm_ids = self.pm.sm_ids_from_sms_str(sm_str)
        seed_models = self.session.query(orm.SeedModel).filter(orm.SeedModel.id.in_(sm_ids)).all()
        return seed_models

    def start_remix_training(self):
        self.start_nni(status_in=orm.RoundStatus.remix_prepped.name,
                       status_out=orm.RoundStatus.remix_training.name,
                       nni_dir=self.pm.nni_training_remix_round(*self.round_handlers),
                       record_to='nni_remix_id')

    def start_remix_evaluation(self):
        self.start_nni(status_in=orm.RoundStatus.remix_training,
                       status_out=orm.RoundStatus.remix_evaluating,
                       nni_dir=self.pm.nni_evaluation_remix_round(*self.round_handlers),
                       record_to='nni_remix_eval_id')

    def start_nni(self, status_in, status_out, nni_dir, record_to):
        """starts nni experiment, copies control files, saves IDs in round 0"""
        assert self.round_handlers[0].round.status.name == status_in, \
            "status mismatch: {} != {}".format(self.round_handlers[0].round.status, status_in)
        sp_args = ['nnictl', 'create', '-c', self.pm.config_yml, '-p', str(self.base_port)]
        print('passing to subprocess: {}\nwith cwd: {}'.format(sp_args, nni_dir))
        subprocess.run(sp_args, cwd=nni_dir)
        # copy control files to nni directory (as usual/for convenience)
        nni_exp_id = self.round_handlers[0].cp_control_files(_from=nni_dir)
        # I'm sure there is a better way, but for now
        if record_to == 'nni_remix_id':
            self.round_handlers[0].round.nni_remix_id = nni_exp_id
        elif record_to == "nni_remix_eval_id":
            self.round_handlers[0].round.nni_remix_eval_id = nni_exp_id
        else:
            raise ValueError(f'unexpected/unhandled string value: {record_to} for "record_to"')
        # record nni_id in db
        for rh in self.round_handlers:
            rh.round.status = status_out
            self.session.add(rh.round)
        self.session.commit()

