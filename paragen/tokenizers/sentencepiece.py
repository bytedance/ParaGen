from typing import List
import os

import sentencepiece as spm

from paragen.tokenizers import AbstractTokenizer, register_tokenizer
from paragen.tokenizers.utils import SPECIAL_SYMBOLS
from paragen.utils.io import mkdir


@register_tokenizer
class SentencePieceTokenizer(AbstractTokenizer):
    """
    SentencePieceTokenizer use sentencepiece lib to do tokenization
    see SentencePiece(https://github.com/google/sentencepiece)

    Args:
        spm_path: sentence piece model path.
    """

    def __init__(self,
                 spm_path=None,
                 add_bos=False,
                 add_eos=False,
                 bos_token='<bos>',
                 eos_token='<eos>',
                 pad_token='<pad>',
                 unk_token='<unk>',):
        super().__init__()
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._sp = None
        try:
            import sentencepiece
            self._sp = sentencepiece.SentencePieceProcessor()
        except ImportError:
            raise ImportError('Please install SentencePiece with: pip install sentencepiece')
        except Exception as e:
            raise Exception("spm error:" + str(e))
        if spm_path is not None:
            status = self._sp.Load(spm_path)
            assert status, "Fail to load spm model: {}".format(spm_path)
        self._bos_token, self._eos_token, self._unk_token, self._pad_token = bos_token, eos_token, unk_token, pad_token

    def __len__(self):
        return len(self._sp)

    @staticmethod
    def learn(input_file, output_path, vocab_size, model_type="bpe", max_sentence_length=4096, user_defined_symbols=None, num_threads=1, character_coverage=0.999):
        """
        learn sentencepiece model from data
        """
        mkdir(output_path)
        default_table = {token: i for i, token in enumerate(SPECIAL_SYMBOLS)}
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=os.path.join(output_path, "spm_{}".format(model_type)),
            vocab_size=vocab_size,
            model_type=model_type,
            unk_id=default_table['<unk>'],
            bos_id=default_table['<bos>'],
            eos_id=default_table['<eos>'],
            pad_id=default_table['<pad>'],
            user_defined_symbols=user_defined_symbols,
            max_sentence_length=max_sentence_length,
            num_threads=num_threads,
            character_coverage=character_coverage,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=True
        )

    def encode(self, x, *args) -> List[int]:
        """
        Encode a textual sentence into a list of index.
        """
        out = [self.bos] if self._add_bos else []
        for ext in x:
            out += self._sp.Encode(ext, add_bos=False, add_eos=self._add_eos)
        return out

    def decode(self, x: List[int]) -> str:
        """
        Decode a list of index back into a textual sentence

        Args:
            x: a list of index
        """
        out = self._sp.Decode(x)
        return out

    def tok(self, *args):
        """
        Tokenize a textual sentence without index mapping.

        Returns:
            - a tokenized textual setnence
        """
        out = [self._bos_token] if self._add_bos else []
        for ext in args:
            out += self._sp.EncodeAsPieces(ext) + ([self._eos_token] if self._add_eos else [])
        return out

    def detok(self, x: str):
        """
        Detokenize a textual sentence without index mapping.

        Args:
            x: a textual sentence

        Returns:
            - a detokenized textual setnence
        """
        return self._tokenize.detok(x)

    @property
    def bos(self):
        return self._sp.bos_id()

    @property
    def eos(self):
        return self._sp.eos_id()

    @property
    def unk(self):
        return self._sp.unk_id()

    @property
    def pad(self):
        return self._sp.pad_id()

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
    def pad_token(self):
        return self._pad_token

    def token2index(self, *args) -> List[int]:
        return [self._sp[x] for x in args]

