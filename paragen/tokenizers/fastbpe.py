from typing import List

from paragen.tokenizers import AbstractTokenizer, register_tokenizer
from paragen.tokenizers.vocabulary import Vocabulary


@register_tokenizer
class FastBPE(AbstractTokenizer):
    """
    Tokenizer use a external tokenization lib with an internal vocabulary

    Args:
        codes: fastBPE codes.
            Generate by ``./fast learnbpe 40000 train.de train.en > codes``
        vocab: fastBPE vocab.
            Generate by ``./fast applybpe train.bpe train codes && ./fast getvocab train.bpe > vocab``
        preserved_tokens: preserved tokens appended to the starts of tokenizer after SPECIAL_SYMBOLS
    """

    def __init__(self,
                 vocab,
                 codes=None,
                 preserved_tokens=None,
                 add_bos=False,
                 add_eos=False,):
        super().__init__()
        self._codes = codes
        self._vocab_path = vocab
        self._preserved_tokens = preserved_tokens
        self._add_bos, self._add_eos = add_bos, add_eos

        self._bpe = None
        self._vocab = None

    def build(self, *args, **kwargs):
        if self._codes is not None:
            import fastBPE
            self._bpe = fastBPE.fastBPE(self._codes, self._vocab)
        self._vocab = Vocabulary(self._vocab_path,
                                 preserved_tokens=self._preserved_tokens,
                                 add_bos=self._add_bos,
                                 add_eos=self._add_eos)

    def __len__(self):
        return len(self._vocab)

    def encode(self, *args: str) -> List[int]:
        """
        Encode a textual sentence into a list of index.
        """
        args = [self._bpe.apply([ext])[0] if self._bpe else ext for ext in args]
        out = self._vocab.encode(args)
        return out

    def decode(self, x: List[int]) -> str:
        """
        Decode a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        x = self._vocab.decode(x)
        x = (x + ' ').replace('@@ ', '').rstrip()
        return x

    def tok(self, *args):
        """
        Tokenize a textual sentence without index mapping.

        Returns:
            - a tokenized textual setnence
        """
        out = [self.bos_token] if self._add_bos else []
        for ext in args:
            x = self._bpe.apply([ext])[0] if self._bpe else ext
            x = self._vocab.encode(x)
            out += x + ([self.eos_token] if self._add_eos else [])
        return out

    def detok(self, x: str):
        """
        Detokenize a textual sentence without index mapping.

        Args:
            x: a textual sentence

        Returns:
            - a detokenized textual sentence
        """
        return (x + ' ').replace('@@ ', '').rstrip()

    def token2index(self, *args) -> List[int]:
        """
        Only map a textual sentence to index

        Returns:
            - an indexed sentence
        """
        out = self._vocab.encode(*args)
        return out

    def index2token(self, x: List[int]) -> str:
        """
        Only map a list of index back into a textual sentence

        Args:
            x: a list of index

        Returns:
            x: a textual sentence
        """
        return self._vocab.decode(x)

    @property
    def bos(self):
        return self._vocab.bos

    @property
    def eos(self):
        return self._vocab.eos

    @property
    def unk(self):
        return self._vocab.unk

    @property
    def pad(self):
        return self._vocab.pad

    @property
    def pad_token(self):
        return self._vocab.pad_token

    @property
    def bos_token(self):
        return self._vocab.bos_token

    @property
    def eos_token(self):
        return self._vocab.eos_token

    @property
    def unk_token(self):
        return self._vocab.unk_token

    @property
    def special_tokens(self):
        return self._vocab.special_tokens
