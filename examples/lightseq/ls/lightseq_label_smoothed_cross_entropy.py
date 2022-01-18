import torch.nn.functional as F

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion
from paragen.utils.ops import search_key
from paragen.utils.runtime import Environment


@register_criterion
class LSLabelSmoothedCrossEntropy(BaseCriterion):
    """
    Label smoothed cross entropy

    Args:
        epsilon: label smoothing rate
    """

    def __init__(self, epsilon: float = 0.1, max_tokens: int = 4096):
        super().__init__()
        self._epsilon = epsilon
        self._max_tokens = max_tokens

        self._padding_idx = None
        self._ls_cross_entropy = None

    def _build(self, model, padding_idx=-1):
        """
        Build a label smoothed cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
            padding_idx: labels of padding_idx are all ignored to computed nll_loss
        """
        from lightseq.training.ops.pytorch.cross_entropy_layer import LSCrossEntropyLayer

        env = Environment()
        self._model = model
        self._padding_idx = padding_idx
        max_tokens = search_key(env.configs['dataloader']['train'], 'max_tokens')
        assert max_tokens, 'max_tokens should be specified when using LSLabelSmoothedCrossEntropy'
        config = LSCrossEntropyLayer.get_config(
            max_batch_tokens=max_tokens,
            padding_idx=self._padding_idx,
            epsilon=self._epsilon,
            fp16=env.fp16,
            local_rank=env.local_rank,
        )
        self._ls_cross_entropy = LSCrossEntropyLayer(config)

    def compute_loss(self, lprobs, target):
        """
        Compute loss from a batch of samples

        Args:
            lprobs: neural network output logits
            target: oracle target for a network input
        Returns:
            - loss for network backward and optimization
            - logging information
        """
        lprobs = F.log_softmax(lprobs, dim=-1)
        assert target.dim() == lprobs.dim() - 1
        # infer task type
        is_classification_task = len(target.size()) == 1
        ntokens = target.ne(self._padding_idx).sum().detach()

        # calculate nll loss
        loss, nll_loss = self._ls_cross_entropy(lprobs, target)
        loss, nll_loss = loss / ntokens, nll_loss / ntokens

        # record logging
        logging_states = {
            'loss': loss.data.item(),
            'nll_loss': nll_loss.data.item(),
            'ntokens': ntokens.data.item(),
        }
        if is_classification_task:
            correct = (lprobs.max(dim=-1)[1] == target).sum().data.item()
            tot = target.size(0)
            logging_states['acc'] = correct / tot
        else:
            logging_states['ppl'] = 2 ** (nll_loss.data.item())

        return loss, logging_states
