import torch
from torch import Tensor
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.layers.learned_positional_embedding import LearnedPositionalEmbedding
from paragen.modules.encoders.layers.transformer_encoder_layer import TransformerEncoderLayer

from .axial_transformer_layer import AxialTransformerLayer


@register_encoder
class TransformerGermlineEncoder(AbstractEncoder):
    """
    TransformerGermlineEncoder is a transformer encoder with germline.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        learn_pos: learning postional embedding instead of sinusoidal one
        return_seed: return with sequence representation
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 n_head=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 return_seed=False,
                 learn_pos=False,
                 normalize_before=False,
                 embed_scale=True,
                 embed_layer_norm=False,
                 max_pos=1024,
                 share_layers=False,
                 layer_type='transformer',
                 layer_reduce_method='last',
                 concat_pos_embed=False,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._activation = activation
        self._return_seed = return_seed
        self._learn_pos = learn_pos
        self._normalize_before = normalize_before
        self._name = name
        self._embed_scale = d_model ** .5 if embed_scale else None
        self._embed_layer_norm = embed_layer_norm
        self._max_pos = max_pos
        self._share_layers = share_layers

        self._special_tokens = None
        self._embed, self._pos_embed, self._embed_norm, self._embed_dropout, self._norm = None, None, None, None, None
        self._layer, self._layers = None, None
        self._pool_seed = None

        self._layer_type = layer_type
        self._layer_reduce_method = layer_reduce_method
        self._concat_pos_embed = concat_pos_embed
        assert layer_type in ('transformer', 'axial_transformer', 'cross_transformer')
        self.EncoderLayer = TransformerEncoderLayer
        if layer_type == 'axial_transformer':
            self.EncoderLayer = AxialTransformerLayer
        elif layer_type == 'cross_transformer':
            raise NotImplementedError

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._embed = embed
        self._special_tokens = special_tokens
        if self._learn_pos:
            self._pos_embed = LearnedPositionalEmbedding(num_embeddings=self._max_pos,
                                                         embedding_dim=self._d_model,)
        else:
            self._pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._embed_norm = nn.LayerNorm(self._d_model) if self._embed_layer_norm else None
        self._embed_dropout = nn.Dropout(self._dropout)
        if self._share_layers:
            self._layer = self.EncoderLayer(d_model=self._d_model,
                                                  nhead=self._n_head,
                                                  dim_feedforward=self._dim_feedforward,
                                                  dropout=self._dropout,
                                                  attention_dropout=self._attention_dropout,
                                                  activation=self._activation,
                                                  normalize_before=self._normalize_before)
            self._layers = [self._layer for _ in range(self._num_layers)]
        else:
            self._layers = nn.ModuleList([self.EncoderLayer(d_model=self._d_model,
                                                                  nhead=self._n_head,
                                                                  dim_feedforward=self._dim_feedforward,
                                                                  dropout=self._dropout,
                                                                  attention_dropout=self._attention_dropout,
                                                                  activation=self._activation,
                                                                  normalize_before=self._normalize_before)
                                          for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def _forward(self, src: Tensor):
        r"""
        Args:
            src: tokens in src side.
              :math:`(BZ, S, L)` where BZ is the batch size, S is the sequence number, L is the source sequence length.

        Outputs:
            - source token hidden representation.
              :math:`(S * L, BZ, D)` where S is the sequence number, L is the source sequence length, BZ is the batch size,
              D is the embedding size.
        """
        batch_size, num_seq, seq_len = src.size()

        if self._concat_pos_embed and self._layer_type == 'transformer':
            src = src.contiguous().view(batch_size, num_seq * seq_len)       # cat germline

        x = self._embed(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            x = x + self._pos_embed(src)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = src.eq(self._special_tokens['pad'])

        if self._layer_type == 'transformer':
            if not self._concat_pos_embed:
                x = x.contiguous().view(batch_size, num_seq * seq_len, -1)       # cat germline
                src_padding_mask = src_padding_mask.contiguous().view(batch_size, num_seq * seq_len)
            x = x.permute(1, 0, 2)
        else:
            x = x.permute(1, 2, 0, 3)
        
        encode_layer_repr = [x]
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)     # (S, L, BZ, D) or (S * L, BZ, D)
            encode_layer_repr.append(x)

        if self._layer_reduce_method == 'last':
            pass
        elif self._layer_reduce_method == 'mean':
            x = torch.stack(encode_layer_repr).mean(0)
        elif self._layer_reduce_method.startswith('layer'):
            idx = int(self._layer_reduce_method.lstrip('layer'))
            x = encode_layer_repr[idx]
        else:
            raise NotImplementedError


        if self._norm is not None:
            x = self._norm(x)

        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask

        return encoder_out

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model
