import itertools
import os
import subprocess

from . import orm
from .paths import PathMaker, robust_4_me_symlink
from .management import RoundHandler, ospj
from .config import Configgy


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
        assert self.round.status.name == orm.RoundStatus.remix_training.name
        # get trial dir
        trial_base = ospj(self.pm.nni_home, self.round_handlers[0].round.nni_remix_id, 'trials')
        trials = os.listdir(trial_base)
        assert len(trials) == len(self.remix_models)
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

    def validation_species(self, training_species):
        all_species = self.session.query(orm.Species).\
            filter(orm.Species.is_quality).all()
        return [x for x in all_species if x not in training_species]

    def setup_remix_data(self):
        """setup remix data combining top trainers from both splits to make final candidate models"""
        for models, remix_dir in zip(self.remix_models, self.remix_dirs):
            remix_sp_t = self.round_handlers[0].seed_model_training_species(models[0]) + \
               self.round_handlers[1].seed_model_training_species(models[1])
            # TODO validation species need to be added after the fact / with knowledge of both splits
            remix_sp_v = self.validation_species(remix_sp_t)
            for sp in remix_sp_t:
                # for fine-tuning w/o changing species, but otherwise to match other adjustments
                robust_4_me_symlink(self.pm.full_h5(sp.name), self.pm.h5_dest(remix_dir, sp.name))

            for sp in remix_sp_v:
                robust_4_me_symlink(self.pm.subset_h5(sp.name), self.pm.h5_dest(remix_dir, sp.name, is_train=False))

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

