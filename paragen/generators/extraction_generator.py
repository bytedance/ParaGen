import torch

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.utils import create_upper_triangular_mask, create_max_segment_mask


@register_generator
class ExtractionGenerator(AbstractGenerator):
    """
    Generator wrap a model with inference algorithms.
    Generator has the same function and interface as model.
    It can be directly exported and used for inference or serving.

    Args:
        path: path to export or load generator
    """

    def __init__(self,
                 path=None,
                 is_regression=False,
                 max_segment_length=None):
        super().__init__(path)
        self._is_regression = is_regression
        self._max_segment_length = max_segment_length

        self._model, self._extraction = None, None
        self._pad = None

    def build_from_model(self, model, pad):
        """
        Build generator from model

        Args:
            model (paragen.models.AbstractModel): a neural model
        """
        self._model = model
        self._extraction = _Extraction(pad, self._max_segment_length)
        self._pad = pad

    def _forward(self, input, *args, **kwargs):
        """
        Infer a sample as model in evaluation mode, and predict results from logits predicted by model
        """
        output = self._model(input, *args, **kwargs)
        start_pos, end_pos = self._extraction(input, output[0], output[1])
        return start_pos, end_pos

    @property
    def model(self):
        return self._model


class _Extraction(torch.nn.Module):
    """
    Extraction methods transform a pair of start and end position to a segment of context.

    Args:
        pad: pad index
        max_segment_length: maximum length for extracted results
    """

    def __init__(self, pad, max_segment_length=None):
        super().__init__()
        self._pad = pad
        self._max_segment_length = max_segment_length

    def forward(self, context, start_logits, end_logits):
        """
        Extract a piece of content from context

        Args:
            context: whole context for extraction
            start_logits: log probability of start position
            end_logits: log probability of end position

        Returns:
            - an extracted sequence of maximum probability
        """
        attention_mask = context.ne(self._pad)
        start_logits = start_logits.masked_fill(~attention_mask, float('-inf'))
        end_logits = end_logits.masked_fill(~attention_mask, float('-inf'))
        batch_size, seqlen = context.size()
        logits = start_logits.unsqueeze(dim=2) + end_logits.unsqueeze(dim=1)
        mask = create_upper_triangular_mask(context)
        if self._max_segment_length:
            max_segment_mask = create_max_segment_mask(context, self._max_segment_length)
            mask = mask & max_segment_mask
        logits = logits.masked_fill(~mask, float('-inf'))
        logits = logits.view(batch_size, seqlen * seqlen)
        _, pos = logits.max(dim=-1)
        start_pos, end_pos = pos // seqlen, pos % seqlen
        return start_pos, end_pos

