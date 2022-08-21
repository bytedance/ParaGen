from functools import reduce
from typing import List
import logging
logger = logging.getLogger(__name__)

from paragen.tokenizers import AbstractTokenizer, register_tokenizer
from paragen.tokenizers.utils import SPECIAL_SYMBOLS
from paragen.utils.io import read_vocab, read_list, UniIO
from paragen.utils.runtime import progress_bar


@register_tokenizer
class Vocabulary(AbstractTokenizer):
    """
    Vocabulary is a naive tokenizer, and just maps tokens to index.

    Args:
        path (str): path of loaded vocabulary
        no_special_symbols (bool): do not append special symbols to token-idx mapping tables
        preserved_tokens: append preserved tokens to token-idx mapping tables
    """

    def __init__(self,
                 path: str,
                 no_special_symbols=False,
                 preserved_tokens=None,
                 add_bos=False,
                 add_eos=False,
                 bos_token='<bos>',
                 eos_token='<eos>',
                 pad_token='<pad>',
                 unk_token='<unk>',):
        super().__init__()
        self._path = path
        self._no_special_symbols = no_special_symbols
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._add_bos, self._add_eos = add_bos, add_eos

        self._length = -1
        self._freq = []
        self._preserved_tokens = []
        self._token2idx = {}
        self._idx2token = {}
        if not self._no_special_symbols:
            self._add_symbols(SPECIAL_SYMBOLS)
        if preserved_tokens is not None:
            if isinstance(preserved_tokens, List):
                self._preserved_tokens = preserved_tokens
            else:
                self._preserved_tokens = read_list(preserved_tokens)
            self._preserved_tokens = [f'<{t}>' for t in self._preserved_tokens]
            self._add_symbols(self._preserved_tokens)
        if self._path:
            logger.info('build vocab from frequency file {}'.format(path))
            self._freq = read_vocab(path)
            self._add_symbols([k for k, _ in self._freq])
            self._length = len(self._token2idx)

    def _add_symbols(self, symbols):
        """
        Add symbols to token-idx mapping tables
        """
        for token in symbols:
            self._token2idx[token] = len(self._token2idx)
            self._idx2token[len(self._idx2token)] = token

    @staticmethod
    def learn(data, output_path, vocab_size=None, threshold=-1):
        """
        learn a vocabulary from data
        """
        def _add_symbol(freq_table, symbol):
            if symbol not in freq_table:
                freq_table[symbol] = 1
            else:
                freq_table[symbol] += 1

        def _add(freq_table, sample):
            if isinstance(sample, str):
                for t in sample.split():
                    _add_symbol(freq_table, t)
            elif isinstance(sample, list):
                return [_add(freq_table, t) for t in sample]
            elif isinstance(sample, dict):
                return {_add(freq_table, key): _add(freq_table, val) for key, val in sample.items()}
            else:
                raise TypeError

        from paragen.datasets import create_dataset
        dataset = create_dataset(data)
        dataset.build()

        freq = {}

        for sample in progress_bar(dataset, streaming=True, desc='Calculating Samples'):
            if isinstance(sample, dict):
                for _, content in sample.items():
                    _add(freq, content)
            else:
                _add(freq, sample)

        freq = [(k, v) for k, v in freq.items() if k not in SPECIAL_SYMBOLS and v > threshold]
        freq.sort(key=lambda x: x[-1], reverse=True)
        if vocab_size:
            freq = freq[:vocab_size]
        logger.info(f'Saving vocabulary (size={len(freq)}) to {output_path}')
        with UniIO(output_path, 'w') as fout:
            for token, freq in freq:
                fout.write('{}\t{}\n'.format(token, freq))

    def encode(self, *args):
        """
        Encode a textual sentence into a list of index.
        """
        if len(args) > 1:
            for x in args:
                assert isinstance(x, str), 'only support multiple string args'
        out = [self._index(ext) for ext in args]
        if self._add_bos:
            out = [out[0]] + [x[1:] for x in out[1:]]
        out = reduce(lambda x, y: x + y, out)
        return out

    def _index(self, token):
        """
        index a (list/dict of) token

        Args:
            token: token, which can be a list, a dict or a str

        Returns:
            index: token index
        """
        def symbol2idx(symbol):
            return self._token2idx[symbol] if symbol in self._token2idx else self.unk

        if isinstance(token, str):
            out = [symbol2idx(t) for t in token.split()]
            if self._add_bos:
                out = [self.bos] + out
            if self._add_eos:
                out = out + [self.eos]
            return out
        elif isinstance(token, list):
            return [self._index(t) for t in token]
        elif isinstance(token, dict):
            return {symbol2idx(key): self._index(val) for key, val in token.items()}
        else:
            raise TypeError

    def decode(self, x):
        """
        Decode a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        return ' '.join(self._token(x))

    def _token(self, index):
        """
        map a (list/dict of) index back to token

        Args:
            index: token, which can be a list, a dict or a str

        Returns:
            index: index
        """
        if isinstance(index, int):
            return self._idx2token[index]
        elif isinstance(index, list):
            return [self._token(i) for i in index]
        elif isinstance(index, dict):
            return {self._token(key): self._token(val) for key, val in index.items()}
        else:
            raise TypeError

    def token2index(self, *args) -> List[int]:
        """
        Only map a textual sentence to index
        """
        return self.encode(*args)

    def index2token(self, x: List[int]) -> str:
        """
        Only map a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        return self.decode(x)

    def __len__(self):
        return self._length

    @property
    def pad(self):
        return self._token2idx[self._pad_token]

    @property
    def bos(self):
        return self._token2idx[self._bos_token]

    @property
    def eos(self):
        return self._token2idx[self._eos_token]

    @property
    def unk(self):
        return self._token2idx[self._unk_token]

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def unk_token(self):
        return self._unk_token

    @property
    def special_tokens(self):
        special_tokens_all = self._preserved_tokens
        if not self._no_special_symbols:
            special_tokens_all += SPECIAL_SYMBOLS
        return {t[1:-1]: self._token2idx[t] for t in special_tokens_all}
