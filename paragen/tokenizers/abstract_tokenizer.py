from typing import List


class AbstractTokenizer:
    """
    Tokenizer provides a tokenization pipeline.
    """

    def __init__(self):
        pass

    def build(self, *args, **kwargs):
        """
        Build tokenizer.
        """
        pass

    def __len__(self):
        raise NotImplementedError

    def encode(self, *args) -> List[int]:
        """
        Encode a textual sentence into a list of index.
        """
        raise NotImplementedError

    def decode(self, x: List[int]) -> str:
        """
        Decode a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        raise NotImplementedError

    def tok(self, *args) -> str:
        """
        Tokenize a textual sentence without index mapping.
        """
        out = []
        for ext in args:
            out += ext
        return out

    def detok(self, x: str) -> str:
        """
        Detokenize a textual sentence without index mapping.

        Args:
            x: a textual sentence
        """
        return x

    def token2index(self, *args) -> List[int]:
        """
        Only map a textual sentence to index
        """
        raise NotImplementedError

    def index2token(self, x: List[int]) -> str:
        """
        Only map a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        raise NotImplementedError

    @staticmethod
    def learn(*args, **kwargs):
        """
        Learn a tokenizer from data set.
        """
        raise NotImplementedError

    @property
    def special_tokens(self):
        return {
            'bos': self.bos,
            'eos': self.eos,
            'pad': self.pad,
            'unk': self.unk
        }

    @property
    def bos(self):
        raise NotImplementedError

    @property
    def eos(self):
        raise NotImplementedError

    @property
    def unk(self):
        raise NotImplementedError

    @property
    def pad(self):
        raise NotImplementedError

    @property
    def bos_token(self):
        raise NotImplementedError

    @property
    def eos_token(self):
        raise NotImplementedError

    @property
    def unk_token(self):
        raise NotImplementedError

    @property
    def pad_token(self):
        raise NotImplementedError
