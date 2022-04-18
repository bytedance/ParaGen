from typing import Dict, List

from paragen.tasks import register_task
from paragen.tasks.seq2seq_task import Seq2SeqTask


@register_task
class HugginfaceSeq2SeqTask(Seq2SeqTask):
    """
    Seq2SeqTask defines overall scope on sequence to sequence task.

    Args:
        src: source key in data dict
        tgt: target key in data dict
        lang: task language
        maxlen: maximum length for sequence
        share_vocab: share source and target vocabulary
        index_only: only do indexing
    """

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def _collate(self, samples: List[Dict]):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        batch = super(HugginfaceSeq2SeqTask, self)._collate(samples)
        if self._infering:
            batch['net_input'] = batch['net_input']['encoder']
        return batch
