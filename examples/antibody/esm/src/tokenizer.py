from typing import List


from paragen.tokenizers import AbstractTokenizer, register_tokenizer
from paragen.utils.runtime import logger

from .esm import pretrained
from .esm.data import Alphabet

@register_tokenizer
class ESMTokenizer(AbstractTokenizer):
    """
    Args:
        tokenizer_name: tokenizer names
    """

    def __init__(self, model_name):
        super().__init__()

        # pretrained_load = getattr(pretrained, model_name)
        # _, alphabet = pretrained_load()
        _, alphabet = pretrained.load_model_and_alphabet(model_name)
        self._tokenizer = alphabet

    @staticmethod
    def learn(*args, **kwargs):
        """
        HuggingfaceTokenizer are used for pretrained model, and is usually directly load from huggingface.
        """
        logger.info('learn vocab not supported for huggingface tokenizer')
        raise NotImplementedError

    def __len__(self):
        return len(self._tokenizer.all_toks)

    def encode(self, input) -> List[int]:
        """
        Encode a textual sentence into a list of index.
        """
        input = input.replace(' ','')
        return self._tokenizer.encode(input)

    def decode(self, tokens) -> str:
        """
        Decode a list of index back into a textual sentence
        """
        return ''.join([self._tokenizer.get_tok(x) for x in tokens])

    def token2index(self, *args) -> List[int]:
        """
        Only map a textual sentence to index
        """
        return self.encode(*args)[1:-1]

    @property
    def bos(self):
        return self._tokenizer.cls_idx

    @property
    def eos(self):
        return self._tokenizer.eos_idx

    @property
    def unk(self):
        return self._tokenizer.unk_idx

    @property
    def pad(self):
        return self._tokenizer.padding_idx

    @property
    def bos_token(self):
        return "<cls>"

    @property
    def eos_token(self):
        return "<eos>"

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return "<pad>"
