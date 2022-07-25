from collections import OrderedDict
from contextlib import contextmanager
import time
import logging
logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter
import torch

from paragen.optim import build_optimizer
from paragen.trainers import AbstractTrainer, register_trainer
from paragen.utils.io import UniIO, mkdir, remove, exists, remove_tree, cp
from paragen.utils.ops import merge_states, deepcopy_on_ref, local_seed
from paragen.utils.runtime import Environment, format_states, str_pipes, progress_bar
from paragen.utils.tensor import half_samples, possible_autocast, possible_scale_loss, to_device, save_ckpt, save_avg_ckpt
from paragen.utils.profiling import profiler


@register_trainer
class Trainer(AbstractTrainer):
    """
    Training scheduler

    Args:
        optimizer: optimizer configurations
        init_epoch (int): initial epoch for training
        init_step (int): initial step for training
        max_epochs (int): training epoch
        max_steps (int): maximum training steps
        validate_interval_epoch (int): epoch-level validation interval
        validate_interval_step (int): step-level validation interval
        log_interval: step-level logging interval
        start_validate_epoch (int): epoch when starting validation
        start_validate_step (int): step when starting validation
        assess_by (str): save the best model by a certain measurement
        assess_reverse: reverse the measurement score when saving the best model
        tensorboard_dir: directory to save tensorboard logs
        save_model_dir: save directory of model checkpoints
        save_best_k: numbers for saving checkpoints with best performances
        save_last_k: numbers for saving last checkpoints
        early_stopping_tolerance: early stop if no better results are derived `tolerance` validation
        force_restart: training from scratch
        restore_path: path to restore checkpoints
        reset_optimizer: reset optimizer without using restored one
        reset_trainer: reset all the states of trainer
        no_best_avg: not save average best checkpoint
    """

    def __init__(self,
                 optimizer,
                 init_epoch=0,
                 init_step=0,
                 max_epochs=None,
                 max_steps=None,
                 validate_interval_epoch=None,
                 validate_interval_step=None,
                 save_steps=None,
                 save_epochs=None,
                 log_interval=500,
                 start_validate_epoch=0,
                 start_validate_step=0,
                 assess_by='criterion',
                 assess_reverse=False,
                 tensorboard_dir=None,
                 save_model_dir=None,
                 save_best_k=10,
                 save_last_k=10,
                 early_stopping_tolerance=None,
                 force_restart=False,
                 restore_path=None,
                 reset_optimizer=False,
                 reset_trainer=False,
                 no_best_avg=True,
                 enable_apex=False,
                 ):
        super().__init__(max_epochs=max_epochs,
                         max_steps=max_steps,
                         validate_interval_epoch=validate_interval_epoch,
                         validate_interval_step=validate_interval_step,
                         save_steps=save_steps,
                         save_epochs=save_epochs,
                         start_validate_epoch=start_validate_epoch,
                         start_validate_step=start_validate_step,
                         log_interval=log_interval
                         )
        self._optimizer_configs = optimizer
        self._tensorboard_dir = tensorboard_dir
        self._save_model_dir = save_model_dir
        self._assess_by = assess_by
        self._assess_reverse = assess_reverse
        self._save_best_k = save_best_k
        self._save_last_k = save_last_k
        self._early_stopping_tolerance = early_stopping_tolerance if early_stopping_tolerance else float('inf')
        self._force_restart = force_restart
        self._restore_path = restore_path
        self._reset_optimizer = reset_optimizer
        self._reset_trainer = reset_trainer
        self._no_best_avg = no_best_avg
        self._enable_apex = enable_apex

        self._best = float('inf') if assess_reverse else float('-inf')
        self._best_info = {}
        self._save_best = []
        self._save_best_avg = []
        self._save_last = []

        self._ori_model, self._model, self._dataloader, self._criterion, self._optimizer = None, None, None, None, None
        self._eval_dataloaders, self._evaluator = None, None
        self._task_callback = None

        self._training_time = 0.
        self._token_count = 0.
        self._progress_bar = None
        self._update_frequency = None
        self._step_cnt, self._tot_step_cnt, self._epoch_cnt = 0, init_step, init_epoch
        self._early_stopping, self._early_stopping_cnt = False, 0
        self._no_progress_bar = None
        self._current_logging_states = OrderedDict()
        self._summary_writer = None

        self._env = Environment()

    def build(self, model, dataloader, criterion, eval_dataloaders=None, evaluator=None, task_callback=None):
        """
        Build trainer from the given configs and components

        Args:
            model: neural model
            dataloader: dataloader of training data
            criterion: criterion for computing objective function
            eval_dataloaders: dataloaders for evaluation
            evaluator: evaluate model with complete generation process in inference
            task_callback: callback to set task states
        """
        self._ori_model = model
        self._dataloader = dataloader
        self._criterion = criterion
        self._eval_dataloaders = eval_dataloaders
        self._evaluator = evaluator
        self._task_callback = task_callback

        self._model, self._optimizer = build_optimizer(model,
                                                       self._optimizer_configs,
                                                       enable_apex=self._enable_apex)
        self._update_frequency = self._optimizer.update_frequency
        self._criterion._model = self._model  # Temporary solution to ddp. Should be fixed in the next version by moving criterion initialization to Trainer.

        self._no_progress_bar = self._env.no_progress_bar or self._env.rank > 0
        if self._tensorboard_dir:
            self._summary_writer = SummaryWriter(self._tensorboard_dir)
        else:
            self._summary_writer = None

        self._possible_restore_checkpoint()

    def _possible_restore_checkpoint(self):
        if self._save_model_dir is not None:
            if self._env.distributed_world > 1:
                self._env.join()
            if exists(self._save_model_dir):
                if self._force_restart:
                    remove_tree(self._save_model_dir)
                    mkdir(self._save_model_dir)
            else:
                mkdir(self._save_model_dir)
            if self._env.distributed_world > 1:
                self._env.join()
            if self._restore_path is None and exists(f'{self._save_model_dir}/last.pt'):
                self._restore_path = f'{self._save_model_dir}/last.pt'

        if self._restore_path is not None:
            logger.info(f'Restore training state from {self._restore_path}')
            with UniIO(self._restore_path, 'rb') as fin:
                state_dict = torch.load(fin, map_location=self._env.device)
                self._restore(state_dict)

        if self._tensorboard_dir:
            if self._force_restart:
                remove_tree(self._tensorboard_dir)
                mkdir(self._tensorboard_dir)
        else:
            mkdir(self._tensorboard_dir)

        if self._env.is_master() \
            and (self._restore_path is not None or self._ori_model.is_pretrained()) \
            and (self._start_validate_epoch == 0 and self._start_validate_step == 0):
            self._eval()

        if self._env.distributed_world > 1:
            self._env.join()

    def train(self):
        """
        Training process
        """
        self.set_mode('train')
        self._tot_step_cnt += 1
        self._epoch_cnt += 1
        while True:
            with self._epoch_context():
                self._epoch_train()
            if self._early_stopping or \
                    (self._max_epochs and self._epoch_cnt > self._max_epochs) or \
                    (self._max_steps and self._tot_step_cnt > self._max_steps):
                break

    def _epoch_train(self):
        """
        Train model in a epoch
        """
        self._task_callback(training=True, infering=False)
        self._progress_bar = progress_bar(self._dataloader)
        samples = []
        for batch in self._progress_bar:
            samples.append(batch)
            if len(samples) == self._update_frequency:
                with self._step_context():
                    self._safe_step(samples)
                samples.clear()
            if self._early_stopping or \
                    (self._max_steps and self._tot_step_cnt > self._max_steps):
                break

    @contextmanager
    def _epoch_context(self):
        """
        For each epoch, reset states before training and update states after training.
        """
        self._loss = 0
        self._nll_loss = 0
        self._step_cnt = 0
        self._dataloader = self._dataloader.reset(self._epoch_cnt)
        self._optimizer.zero_grad()

        yield

        if self._save_epochs and self._epoch_cnt % self._save_epochs == 0:
            if self._env.is_master():
                if self._save_model_dir:
                    self._save_last_model()
            if self._env.distributed_world > 1:
                self._env.join()

        # check if doing evaluation
        if self._validate_interval_epoch and \
                self._epoch_cnt >= self._start_validate_epoch and \
                self._epoch_cnt % self._validate_interval_epoch == 0:
            if self._env.is_master():
                eval_states = self._eval()
                if self._save_model_dir and self._assess_by in eval_states:
                    self._save_best_model(**eval_states)

                if self._tensorboard_dir:
                    self._update_tensorboard('eval', eval_states)
            if self._env.distributed_world > 1:
                self._env.join()

        self._epoch_cnt += 1

    def _safe_step(self, samples):
        """
        Safely step a batch of samples

        Args:
            samples: a set of batches
        """
        finished = False
        is_dummy = False
        cached_samples = deepcopy_on_ref(samples)
        while not finished:
            try:
                self._current_logging_states = self._step(samples, is_dummy)
                finished = True
                break
            except RuntimeError as e:
                error_info = str(e)
                error_info = error_info.split('\n')[0]
                logger.warning(error_info)
                oom = 'out of memory' in error_info
                self._current_logging_states = {}
                if not oom:
                    raise e
            if oom and self._env.device == 'cuda':
                torch.cuda.empty_cache()
                self._optimizer.zero_grad()
                with local_seed(self._tot_step_cnt):
                    samples, is_dummy = half_samples(cached_samples)
                cached_samples = deepcopy_on_ref(samples)

    def _step(self, samples, is_dummy=False):
        """
        Train a set of batches with only one gradient update.

        Args:
            samples: a set of batches

        Returns:
            logging_states: states to display in progress bar
        """
        self._optimizer.zero_grad()
        samples = to_device(samples, device=self._env.device)
        logging_states = OrderedDict()
        for i, batch in enumerate(samples):
            self._ori_model.reset(mode='train')
            with profiler.timeit("forward"):
                if self._enable_apex:
                    loss, logging_state = self._forward_loss(batch)
                else:
                    with possible_autocast():
                        loss, logging_state = self._forward_loss(batch)
            with profiler.timeit("backward"):
                self._backward_loss(loss)
            logging_states = merge_states(logging_states,
                                          logging_state,
                                          weight=1./(i+1.))
        if is_dummy:
            logger.info('dummy batch detected! set gradients to zero!')
            self._optimizer.multiply_grads(0.)
        with profiler.timeit("optimizer"):
            self._optimizer.step()

        return logging_states

    def _forward_loss(self, samples):
        """
        Forward neural model and compute the loss of given samples

        Args:
            samples: a batch of samples

        Returns:
            - derived loss as torch.Tensor
            - states for updating log
        """
        loss, logging_states = self._criterion(**samples)
        return loss, logging_states

    def _backward_loss(self, loss):
        """
        Backward loss and compute gradients on weights

        Args:
            loss: derived loss
        """
        if self._enable_apex:
            from apex import amp
            with amp.scale_loss(loss, self._optimizer.optimizer) as scaled_loss:
                scaled_loss.backward()
                if self._env.distributed_world > 1:
                    self._optimizer.optimizer.synchronize()
        else:
            scaled_loss = possible_scale_loss(loss)
            scaled_loss.backward()
            if self._env.distributed_world > 1 and self._env.fp16 and self._env.distributed in ['horovod', 'hvd']:
                self._optimizer.optimizer.synchronize()

    @contextmanager
    def _step_context(self):
        """
        For each step, reset states before training and update states after training.
        """
        start_time = time.time()
        profiler.cycle_start()

        yield

        profiler.cycle_end()
        end_time = time.time()
        self._training_time += end_time - start_time
        if 'ntokens' in self._current_logging_states:
            real_ntokens = self._current_logging_states['ntokens'] * self._env.distributed_world * self._update_frequency
            self._token_count += real_ntokens
            self._current_logging_states['ntokens'] = real_ntokens

        # update logging on tqdm
        self._update_logging()
        if self._env.is_master() and self._tensorboard_dir:
            self._update_tensorboard('train', self._current_logging_states)
        if not self._no_progress_bar:
            logging_states = format_states(self._current_logging_states)
            self._progress_bar.set_postfix(ordered_dict=logging_states,
                                           refresh=False)
        if self._tot_step_cnt % self._log_interval == 0:
            logger.info(str_pipes(format_states(self._current_logging_states)))

        if self._save_steps and self._tot_step_cnt % self._save_steps == 0:
            if self._env.is_master():
                if self._save_model_dir:
                    self._save_last_model()
            if self._env.distributed_world > 1:
                self._env.join()

        # check if doing evaluation
        if self._validate_interval_step and \
                self._tot_step_cnt >= self._start_validate_step and \
                self._tot_step_cnt % self._validate_interval_step == 0:
            if self._env.is_master():
                eval_states = self._eval()
                if self._save_model_dir and self._assess_by in eval_states:
                    self._save_best_model(**eval_states)

                if self._tensorboard_dir:
                    self._update_tensorboard('eval', eval_states)
            if self._env.distributed_world > 1:
                self._env.join()

        self._step_cnt += 1
        self._tot_step_cnt += 1

        self._dataloader.step_update(self._tot_step_cnt)
        self._criterion.step_update(self._tot_step_cnt)
        # lazy update for saving computation
        self._optimizer.step_update(self._tot_step_cnt)

    def _update_logging(self):
        """
        update global states with states at one step
        """
        logging_states = OrderedDict()
        logging_states['epochs'] = self._epoch_cnt
        logging_states['steps'] = self._tot_step_cnt
        logging_states['lr'] = self._optimizer.lr
        if self._token_count > 0:
            logging_states['wps'] = self._token_count / self._training_time
        self._current_logging_states = merge_states(logging_states, self._current_logging_states, 0)

    def _eval(self):
        """
        Evaluate the model, and save model & tensorboard log
        Two evaluation are performed:
            1. eval with training objective
            2. eval with a given evaluator, which usually involves inference algorithms to generate final results.
        """
        assert self._env.is_master(), "only master process is allowed to perform evaluation"
        logger.info(str_pipes(format_states(self._current_logging_states)))
        eval_states = {}
        if self._eval_dataloaders:
            scores = self._eval_by_criterion()
            eval_states.update(scores)
        if self._evaluator:
            scores = self._eval_by_evaluator()
            eval_states.update(scores)

        if (self._assess_reverse and self._best > eval_states[self._assess_by]) \
                or (not self._assess_reverse and self._best < eval_states[self._assess_by]):
            self._best = eval_states[self._assess_by]
            self._best_info = {f'best.{key}': value for key, value in eval_states.items()}
        logger.info(str_pipes(format_states(self._best_info)))

        self.set_mode('train')
        return eval_states

    def _eval_by_criterion(self):
        """
        Eval model with training objective

        Returns:
            evaluation results
        """
        self.set_mode('valid')
        scores = {}
        tot_loss, cnt = 0, 0
        self._eval_dataloaders = {
            data_name: dataloader.reset()
            for data_name, dataloader in self._eval_dataloaders.items()
        }
        for data_name, dataloader in self._eval_dataloaders.items():
            logger.info(f'eval on {data_name} dataset by criterion {self._criterion.__class__.__name__}')
            loss = self._eval_dataset_by_criterion(dataloader)
            tot_loss += loss
            cnt += 1
            scores[f'{data_name}.criterion'] = loss
        scores['criterion'] = tot_loss / cnt
        return scores

    def _eval_dataset_by_criterion(self, dataloader):
        """
        Evaluate model on a dataloader with training criterion

        Args:
            dataloader: a evaluation dataloader

        Returns:
            loss: average development loss on the given dataloder
        """
        dev_loss, dev_cnt = 0, 0
        logging_avg_states = {}
        with torch.no_grad():
            pb = progress_bar(dataloader)
            for samples in pb:
                self._ori_model.reset(mode='valid')
                samples = to_device(samples, device=self._env.device)
                with possible_autocast():
                    loss, logging_states = self._forward_loss(samples)
                dev_loss += loss.data.item()
                dev_cnt += 1
                logging_avg_states = merge_states(logging_avg_states, logging_states, 1. / dev_cnt)
                if not self._no_progress_bar:
                    pb.set_postfix(ordered_dict=format_states(logging_avg_states),
                                   refresh=False)
        logger.info(str_pipes(format_states(logging_avg_states)))
        return dev_loss / dev_cnt

    def _eval_by_evaluator(self):
        """
        Eval with a given evaluator, which usually involves inference algorithms to generate final results.

        Returns:
            scores: score produced by evaluator
        """
        self.set_mode('eval')
        scores = self._evaluator.eval()
        return scores

    def _save_last_model(self):
        logger.info('Saving the last model starts.')
        assert self._env.is_master(), "only master process is allowed to save models"
        name = f'updates-{self._tot_step_cnt}.epochs-{self._epoch_cnt}'

        path = f'{self._save_model_dir}/last.{name}.pt'
        self._save_last.append(path)

        while len(self._save_last) > self._save_last_k:
            rm_path = self._save_last.pop(0)
            logger.info(f'remove last checkpoint at {rm_path}')
            remove(rm_path)

        logger.info(f'save last k checkpoint to {path}')
        state_dict = self.state_dict()
        save_ckpt(state_dict, path)

        logger.info(f'copy the last checkpoint to {self._save_model_dir}/last.pt')
        cp(path, f'{self._save_model_dir}/last.pt')

        assert len(self._save_last) <= self._save_last_k, f'size of save_last must be <= save_last_k, {self._save_last}'
        logger.info('Saving Done.')

    def _save_best_model(self, **kwargs):
        """
        save model according to evaluation results

        Args:
            **kwargs: saving information for compute checkpoint name
        """
        logger.info('Saving the best model starts.')
        assert self._env.is_master(), "only master process is allowed to save models"
        name_step = f'updates-{self._tot_step_cnt}.epochs-{self._epoch_cnt}'
        name_metric = '.'.join(
            [('{}-{:.4f}' if isinstance(v, float) else '{}-{}').format(k, v)
             for k, v in kwargs.items() if self._assess_by in k])
        name = f'{name_step}.{name_metric}'

        state_dict = self.state_dict()

        path = f'{self._save_model_dir}/best.{name}.pt'
        avg_path = f'{self._save_model_dir}/best_avg.{name}.pt' if not self._no_best_avg else None
        score = kwargs[self._assess_by]
        logger.info(f'{len(self._save_best)}/{self._save_best_k} best checkpoints found!')
        if len(self._save_best) < self._save_best_k:
            logger.info(f'save best k checkpoint to {path}')
            save_ckpt(state_dict, path)
            self._save_best.append((path, score))
            self._save_best.sort(key=lambda x: x[-1], reverse=not self._assess_reverse)
            if avg_path is not None:
                logger.info(f'save best average k checkpoint to {avg_path}')
                save_avg_ckpt(self._save_last, avg_path)
                self._save_best_avg.append((avg_path, score))
                self._save_best_avg.sort(key=lambda x: x[-1], reverse=not self._assess_reverse)
        else:
            self._save_best.append((path, score))
            self._save_best.sort(key=lambda x: x[-1], reverse=not self._assess_reverse)
            if avg_path is not None:
                self._save_best_avg.append((avg_path, score))
                self._save_best_avg.sort(key=lambda x: x[-1], reverse=not self._assess_reverse)

            if self._save_best[-1][0] != path:
                logger.info(f'better checkpoint appears')
                logger.info(f'save best k checkpoint to {path}')
                save_ckpt(state_dict, path)
                logger.info(f'remove (no longer) best k checkpoint at {self._save_best[-1][0]}')
                remove(self._save_best[-1][0])
                if avg_path is not None:
                    logger.info(f'save best average k checkpoint to {avg_path}')
                    save_avg_ckpt(self._save_last, avg_path)
                    logger.info(f'remove (no longer) best average k checkpoint at {self._save_best_avg[-1][0]}')
                    remove(self._save_best_avg[-1][0])

            self._save_best.pop(-1)
            if avg_path is not None:
                self._save_best_avg.pop(-1)

        if self._save_best[0][1] == score:
            logger.info(f'new best checkpoint appears')
            self._early_stopping_cnt = 0
            logger.info(f'copy the best checkpoint to {self._save_model_dir}/best.pt')
            cp(path, f'{self._save_model_dir}/best.pt')
            if avg_path is not None:
                logger.info(f'copy the best average checkpoint to {self._save_model_dir}/best_avg.pt')
                cp(avg_path, f'{self._save_model_dir}/best_avg.pt')
        else:
            self._early_stopping_cnt += 1
            if self._early_stopping_cnt > self._early_stopping_tolerance:
                self._early_stopping = True
                return

        assert len(self._save_best) <= self._save_best_k, f'size of save_best must be <= save_best_k, {self._save_best}'
        logger.info('Saving Done.')

    def _update_tensorboard(self, name, states):
        """
        update tensorboard with training/eval states

        Args:
            name: tensorboard name
            states: training/eval states
        """
        for key, val in states.items():
            self._summary_writer.add_scalar(f'{name}.{key}', val, self._tot_step_cnt)

    def set_mode(self, mode):
        """
        Switch mode ['train', 'valid', 'infer'] of a trainer

        Args:
            mode: trainer mode, ['train', 'valid', 'infer']
        """
        self._ori_model.reset(mode)
        self._task_callback(training=(mode == 'train'),
                            infering=(mode == 'infer'))
        if mode == 'train':
            self._model.train()
            self._criterion.train()
        else:
            self._model.eval()
            self._criterion.eval()

    def state_dict(self):
        model_state_dict = self._model.state_dict()
        optimizer_state_dict = self._optimizer.state_dict()
        trainer_state_dict = {
            'best': self._best,
            'best_info': self._best_info,
            'save_best': self._save_best,
            'save_best_avg': self._save_best_avg,
            'save_last': self._save_last,
            'step_cnt': self._step_cnt,
            'tot_step_cnt': self._tot_step_cnt,
            'epoch_cnt': self._epoch_cnt,
            'early_stopping': self._early_stopping,
            'early_stopping_cnt': self._early_stopping_cnt,
        }
        return {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'trainer': trainer_state_dict
        }

    def _restore(self, state_dict):
        self._model.load_state_dict(state_dict['model'])
        logger.info('Successfully restore model')
        if not self._reset_optimizer and 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])
            logger.info('Successfully restore optimizer')

        if not self._reset_trainer and 'trainer' in state_dict:
            trainer_state_dict = state_dict['trainer']
            self._best = trainer_state_dict['best']
            self._best_info = trainer_state_dict['best_info']
            self._save_best = trainer_state_dict['save_best']
            self._save_best_avg = trainer_state_dict['save_best_avg']
            self._save_last = trainer_state_dict['save_last']
            self._step_cnt = trainer_state_dict['step_cnt']
            self._tot_step_cnt = trainer_state_dict['tot_step_cnt']
            self._epoch_cnt = trainer_state_dict['epoch_cnt']
            self._early_stopping = trainer_state_dict['early_stopping']
            self._early_stopping_cnt = trainer_state_dict['early_stopping_cnt']
            logger.info('Successfully restore trainer')