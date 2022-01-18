import torch.nn as nn
import torch.nn.functional as F

from paragen.models import register_model
from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel
from paragen.modules.decoders import AbstractDecoder
from paragen.modules.encoders import AbstractEncoder
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.utils import create_time_mask, create_source_target_modality


@register_model
class TorchTransformer(AbstractEncoderDecoderModel):
    """
    TorchTransformer is tranformer implemented in torch

    Args:
        d_model: feature embedding
        nhead: head number in multihead attention
        num_encoder_layers: number of encoder layers
        num_decoder_layers: number of decoder layers
        dim_feedforward: inner dimension in feed-forward network
        dropout: dropout rate
        activation: activation function name
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        path: path to restore model
    """

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 share_embedding='decoder-input-output',
                 path=None):
        super().__init__(path=path)
        self._d_model = d_model
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._share_embedding = share_embedding
        self._embed_scale = d_model ** 0.5

        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._src_embed, self._tgt_embed, self._tgt_out_weight = None, None, None
        self._src_pos_embed, self._tgt_pos_embed = None, None
        self._embed_dropout = nn.Dropout(self._dropout)
        self._transformer = None
        self._mode = 'train'

    def build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build encoder-decoder model

        Args:
            src_vocab_size: vocabulary size at source sitde
            tgt_vocab_size: vocabulary size at target sitde
            src_special_tokens: special tokens in source vocabulary
            tgt_special_tokens: special tokens in target vocabulary
        """
        self._src_special_tokens = src_special_tokens
        self._tgt_special_tokens = tgt_special_tokens
        self._create_modality(src_vocab_size=src_vocab_size,
                              tgt_vocab_size=tgt_vocab_size,
                              src_padding_idx=src_special_tokens['pad'],
                              tgt_padding_idx=tgt_special_tokens['pad'])
        self._transformer = nn.Transformer(self._d_model,
                                           self._nhead,
                                           self._num_encoder_layers,
                                           self._num_decoder_layers,
                                           self._dim_feedforward,
                                           self._dropout,
                                           self._activation)

        self._encoder = _Encoder(self._transformer.encoder, padding_idx=src_special_tokens['pad'], map_fn=self.map)
        self._decoder = _Decoder(self._transformer.decoder, map_fn=self.map)

    def _create_modality(self, src_vocab_size, tgt_vocab_size, src_padding_idx, tgt_padding_idx):
        """
        Create modality, including token and positional embedding at source and target side.

        Args:
            src_vocab_size: vocabulary size at source side
            tgt_vocab_size: vocabulary size at target side
            src_padding_idx: padding_idx in source vocabulary
            tgt_padding_idx: padding_idx in target vocabulary
        """
        self._src_embed, self._tgt_embed, self._tgt_out_weight = create_source_target_modality(
            d_model=self._d_model,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_padding_idx=src_padding_idx,
            tgt_padding_idx=tgt_padding_idx,
            share_embedding=self._share_embedding
        )

        self._src_pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._tgt_pos_embed = SinusoidalPositionalEmbedding(self._d_model)

    def forward(self, src, tgt):
        """
        Compute output with neural input

        Args:
            src: source sequence
                :math:`(N, S)` where N is batch size and S is source sequence length
            tgt: previous tokens at target side, which is a time-shifted target sequence in training
                :math:`(N, T)` where N is batch size and T is target sequence length

        Returns:
            - log probability of next token at target side
        """
        src, src_padding_mask = self.map(src, self._src_embed, self._src_pos_embed, self._src_padding_idx)
        tgt, tgt_padding_mask = self.map(tgt, self._tgt_embed, self._tgt_pos_embed, self._tgt_padding_idx)
        tgt_mask = create_time_mask(tgt)
        out = self._transformer(src=src.transpose(0, 1),
                                tgt=tgt.transpose(0, 1),
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask).transpose(0, 1)
        logits = F.linear(out, self._tgt_out_weight)
        logits = F.log_softmax(logits, dim=-1)
        return logits

    def map(self, tokens, embed, pos_embed, padding_idx):
        """
        Forward embedding look-up operation

        Args:
            tokens: token index
                :math:`(N, T)` where N is batch size and T is sequence length.
            embed: embedding module
            pos_embed: position embedding module
            padding_idx: padding index to ignore

        Returns:
            - feature embedding of tokens
                :math:`(N, T, E)` where N is batch size, T is sequence length and E is feature dimension.
            - padding mask of tokens
                :math:`(N, T)` where N is batch size and T is sequence length
        """
        x = embed(tokens) * self._embed_scale + pos_embed(tokens)
        padding_mask = tokens.eq(padding_idx)
        x = self._embed_dropout(x)
        return x, padding_mask


class _Encoder(AbstractEncoder):
    """
    Inner encoder class for TorchTransformers

    Args:
        encoder: encoder in torch-implemented transformers
        padding_idx: padding index to ignore
        map_fn: embedding lookup function
    """

    def __init__(self, encoder, padding_idx, map_fn):
        super().__init__()
        self._encoder = encoder
        self._padding_idx = padding_idx
        self._map_fn = map_fn

    def forward(self, src):
        """
        Compute output with encoder

        Args:
            src: source sequence
                :math:`(N, S)` where N is batch size and S is source sequence length

        Returns:
            - source features:
                :math:`(S, N, E)` where S is source sequence length, N is batch size and E is feature dimension.
            - source padding mask:
                :math:`(N, S)` where N is batch size and S is source sequence length.
        """
        src, src_padding_mask = self._map_fn(src, self._src_embed, self._src_pos_embed, self._src_padding_idx)
        src = src.transpose(0, 1)
        src = self._transformer.encoder(src,
                                        src_key_padding_mask=src_padding_mask)
        return src, src_padding_mask


class _Decoder(AbstractDecoder):
    """
    Inner decoder class for TorchTransformers

    Args:
        decoder: decoder in torch-implemented transformers
        map_fn: embedding lookup function
    """

    def __init__(self, decoder, map_fn):
        super().__init__()
        self._decoder = decoder
        self._map_fn = map_fn

    def forward(self, tgt, memory, memory_padding_mask):
        """
        Compute output with encoder

        Args:
            tgt: previous tokens at target size
                :math:`(N, T)` where N is batch size and T is target sequence length
            memory: feature memory for attention
                :math:`(S, N, E)` where S is source sequence length, N is batch size and E is feature dimension.
            memory_padding_mask: padding mask over feature memory
                :math:`(N, S)` where N is batch size and S is source sequence length.

        Returns:
            - log probability of next token at target side
                :math:`(N, T, V)` where N is batch size, T is target sequence length
                and V is target vocabulary size.
        """
        tgt, tgt_padding_mask = self._map_fn(tgt, self._tgt_embed, self._tgt_pos_embed, self._tgt_padding_idx)
        tgt_mask = create_time_mask(tgt)
        tgt = tgt.transpose(0, 1)
        tgt = self._transformer.decoder(tgt, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_padding_mask, )
        tgt = tgt.transpose(0, 1)
        logits = F.linear(tgt, self._tgt_out_weight)
        logits = F.log_softmax(logits, dim=-1)
        return logits
