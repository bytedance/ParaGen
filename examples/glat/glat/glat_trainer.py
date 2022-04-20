import torch
from paragen.trainers.trainer import Trainer
from paragen.trainers import register_trainer


@register_trainer
class GLATTrainer(Trainer):
    """
    Trainer with glancing strategy

    Args:
        minus_p; glancing minus_p
    """
    def __init__(self, max_context_p=0.5, minus_p=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_context_p = max_context_p
        self._minus_p = minus_p

        self._generator = None
        self._src_special_tokens, self._tgt_special_tokens = None, None

    def build(self, generator, src_special_tokens, tgt_special_tokens, **kwargs):
        """
        Build trainer from the given configs and components

        Args:
            generator: neural model with inference algorithm
            src_special_tokens: special_tokens at source side
            tgt_special_tokens: special_tokens at target side
        """
        self._generator = generator
        self._src_special_tokens = src_special_tokens
        self._tgt_special_tokens = tgt_special_tokens

        super().build(**kwargs)

    def _forward_loss(self, samples):
        """
        Train one batch of samples

        Args:
            samples: a batch of samples
        Returns:
            logging_states: states to display in progress bar
        """
        self._model.reset(mode='train')
        self._model.set_seed(self._tot_step_cnt)
        glancing_output = self._generator(**samples['net_input'])

        glancing_target = samples.pop('glancing_target')
        masked_target, fusing_target_mask = self.glancing(
            glancing_output,
            **glancing_target
        )
        samples['net_input']['target'] = glancing_target['target']
        samples['net_input']['fusing_target_mask'] = fusing_target_mask
        samples['net_output']['token']['target'] = masked_target

        loss, logging_states = self._criterion(**samples)

        if torch.isnan(loss).any():
            logging_states = {}
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
        with torch.no_grad():
            neq_mask = target.ne(prediction)
            neq_mask = neq_mask.masked_fill(target_padding_mask, False)
            neq_cnts = neq_mask.sum(dim=-1)

            bsz, seqlen = target.size()
            seqlen_i = (~target_padding_mask).sum(dim=-1)
            context_p = self._max_context_p - self._minus_p * min(1, max(0, self._tot_step_cnt / self._max_steps))
            fusing_target_num = (neq_cnts.float() * context_p).long()
            fusing_target_mask = torch.ones_like(prediction)
            for li in range(bsz):
                if fusing_target_num[li] > 0:
                    index = torch.randperm(seqlen_i[li])[:fusing_target_num[li]].to(seqlen_i)
                    fusing_target_mask[li].scatter_(dim=0, index=index, value=0)
            fusing_target_mask = fusing_target_mask.eq(0)

            fusing_target_mask = fusing_target_mask.masked_fill(target_padding_mask, False)
            target = target.masked_fill(fusing_target_mask, self._tgt_special_tokens['pad'])
        return target.detach(), fusing_target_mask.detach()



