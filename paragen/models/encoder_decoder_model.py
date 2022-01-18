from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel
from paragen.modules.decoders import create_decoder
from paragen.modules.encoders import create_encoder
from paragen.modules.utils import create_source_target_modality


class EncoderDecoderModel(AbstractEncoderDecoderModel):
    """
    EncoderDecoderModel defines overall encoder-decoder architecture.

    Args:
        encoder: encoder configurations to build an encoder
        decoder: decoder configurations to build an decoder
        d_model: feature embedding
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None):
        super().__init__(path=path)
        self._encoder_config, self._decoder_config = encoder, decoder
        self._d_model = d_model
        self._share_embedding = share_embedding
        self._path = path

    def _build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build encoder-decoder model

        Args:
            src_vocab_size: vocabulary size at source sitde
            tgt_vocab_size: vocabulary size at target sitde
            src_special_tokens: special tokens in source vocabulary
            tgt_special_tokens: special tokens in target vocabulary
        """
        src_embed, tgt_embed, tgt_out_proj = create_source_target_modality(
            d_model=self._d_model,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_padding_idx=src_special_tokens['pad'],
            tgt_padding_idx=tgt_special_tokens['pad'],
            share_embedding=self._share_embedding
        )
        self._encoder = create_encoder(self._encoder_config)
        self._decoder = create_decoder(self._decoder_config)
        self._encoder.build(embed=src_embed, special_tokens=src_special_tokens)
        self._decoder.build(embed=tgt_embed,
                            special_tokens=tgt_special_tokens,
                            out_proj=tgt_out_proj)

    def reset(self, mode, *args, **kwargs):
        """
        Switch mode and reset internal states

        Args:
            mode: running mode
        """
        self._mode = mode
        self._encoder.reset(mode, *args, **kwargs)
        self._decoder.reset(mode, *args, **kwargs)

    def set_cache(self, cache):
        """
        Set internal cache with outside one

        Args:
            cache: neural model cache states
        """
        if 'encoder' in cache:
            self._encoder.set_cache(cache['encoder'])
        elif 'decoder' in cache:
            self._decoder.set_cache(cache['decoder'])
        else:
            raise LookupError

    def get_cache(self):
        """
        Retrieve internal cache

        Returns:
            - internal cache
        """
        return {
            'encoder': self._encoder.get_cache(),
            'decoder': self._decoder.get_cache()
        }
