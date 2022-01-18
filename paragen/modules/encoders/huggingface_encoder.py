from torch import Tensor
from transformers import AutoConfig, AutoModel

from paragen.modules.encoders import AbstractEncoder, register_encoder


@register_encoder
class HuggingFaceEncoder(AbstractEncoder):
    """
    HuggingFaceEncoder is a wrapped encoder from huggingface pretrained models

    Args:
        pretrained_model: name of pretrained model, see huggingface for supported models
        freeze: freeze pretrained model in training
        return_seed: return with sequence representation
        name: encoder name
    """

    def __init__(self,
                 pretrained_model,
                 freeze=False,
                 return_seed=False,
                 name=None):
        super().__init__(name=name)
        self._pretrained_model_name = pretrained_model
        self._freeze = freeze
        self._return_seed = return_seed

        self._special_tokens = None
        self._configs = None
        self._huggingface_model = None

    def build(self, special_tokens=None, vocab_size=None):
        """
        Build computational graph

        Args:
            special_tokens: special_tokens: special tokens defined in vocabulary
            vocab_size: vocabulary size of embedding
        """
        self._special_tokens = special_tokens
        self._configs = AutoConfig.from_pretrained(self._pretrained_model_name)
        self._huggingface_model = AutoModel.from_config(self._configs)
        if self._freeze:
            self.freeze_params()

        assert self._configs.vocab_size == vocab_size

    def freeze_params(self):
        """
        Freeze parameters of pretrained model
        """
        for param in self._huggingface_model.base_model.parameters():
            param.requires_grad = False

    def _forward(self, text: Tensor):
        r"""
        Args:
            text: tokens in src side.
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Returns:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        padding_mask = text.eq(self._padding_idx)
        model_out = self._huggingface_model(text, ~padding_mask)
        try:
            x, seed = model_out['last_hidden_state'], model_out['pooler_output']
        except:
            x, seed = model_out
        finally:
            x = x.transpose(0, 1)
            if self._return_seed:
                return x, padding_mask, seed
            else:
                return x, padding_mask

    @property
    def out_dim(self):
        return self._configs.hidden_size

