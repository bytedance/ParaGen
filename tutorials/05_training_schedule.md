ParaGen allows customization for training scheduling by re-implementing a `Trainer`.
A ParaGen `Trainer` consists of following functions:
- `train`, `epoch_train`, `step`: generally schedule training at different granularity;
- `possible_restore_checkpoint`, `save_model`: restore and save a checkpoint;
- `eval`, `eval_by_criterion`, `eval_by_evaluator`: evaluate current model with loss or customized evaluation standard;
- `update_logging`: create logging information displayed on terminal;
- `forward_loss`: compute loss for neural model on current batches;
- `backward_loss`: backward gradients with respect to loss;
- `step`: update batches by `forward`, `backward` and `optimizer.step`.

# A practical example to customize trainer

Here we give a example of [glancing training](../examples/glat/glat/glat_trainer.py):
```python
from paragen.trainers.trainer import Trainer
from paragen.trainers import register_trainer


@register_trainer
class GLATTrainer(Trainer):
    def __init__(self, minus_p=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._minus_p = minus_p

        self._generator = None
        self._src_special_tokens, self._tgt_special_tokens = None, None

    def build(self, generator, src_special_tokens, tgt_special_tokens, **kwargs):
        super().build(**kwargs)
        self._generator = generator
        self._src_special_tokens = src_special_tokens
        self._tgt_special_tokens = tgt_special_tokens

    def _forward_loss(self, samples):
        self._model.reset(mode='train')
        glancing_output = self._generator(**samples['net_input'])

        glancing_target = samples.pop('glancing_target')
        masked_target, fusing_target_mask = self.glancing(glancing_output, **glancing_target)
        samples['net_input']['target'] = glancing_target['target']
        samples['net_input']['fusing_target_mask'] = fusing_target_mask
        samples['net_output']['token']['target'] = masked_target

        loss, logging_states = self._criterion(**samples)

        return loss, logging_states

    def glancing(self, prediction, target, target_padding_mask):
        """
        Glancing strategy

        Args:
            prediction: predicted results
            target: target output
            target_padding_mask: padding_mask for target output

        Returns:
            - training target with (probabilistically) unsure tokens as pad
            - (probabilistically) unsure tokens mask
        """
        pass # glancing strategy
```

Typically, the rewrited trainer is a subclass of trainer, which reuse save & restore, logging and evaluation codes.
In most cases, `_forward_loss` is main function for modification.
Extra configurations are added to `__init__` to receive new arguments.
Then `build` function is able accept new resources from a task.
Finally, we fully customize `forward_loss` to change the loss computation.
