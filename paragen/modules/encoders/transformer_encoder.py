from torch import Tensor
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.layers.learned_positional_embedding import LearnedPositionalEmbedding
from paragen.modules.encoders.layers.transformer_encoder_layer import TransformerEncoderLayer


@register_encoder
class TransformerEncoder(AbstractEncoder):
    """
    TransformerEncoder is a transformer encoder.

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
                 keep_seed_in_hidden=False,
                 learn_pos=False,
                 normalize_before=False,
                 embed_scale=True,
                 embed_layer_norm=False,
                 max_pos=1024,
                 share_layers=False,
                 position_emb_post_mask=False,
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
        self._keep_hidden_in_hidden = keep_seed_in_hidden
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
        self._position_emb_post_mask = position_emb_post_mask

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
                                                         embedding_dim=self._d_model,
                                                         padding_idx=special_tokens['pad'],
                                                         post_mask=self._position_emb_post_mask)
        else:
            self._pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._embed_norm = nn.LayerNorm(self._d_model) if self._embed_layer_norm else None
        self._embed_dropout = nn.Dropout(self._dropout)
        if self._share_layers:
            self._layer = TransformerEncoderLayer(d_model=self._d_model,
                                                  nhead=self._n_head,
                                                  dim_feedforward=self._dim_feedforward,
                                                  dropout=self._dropout,
                                                  attention_dropout=self._attention_dropout,
                                                  activation=self._activation,
                                                  normalize_before=self._normalize_before)
            self._layers = [self._layer for _ in range(self._num_layers)]
        else:
            self._layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
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
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        x = self._embed(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            x = x + self._pos_embed(src)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = src.eq(self._special_tokens['pad'])
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)

        if self._norm is not None:
            x = self._norm(x)

        if self._return_seed:
            shift = 0 if self._keep_hidden_in_hidden else 1
            encoder_out = x[shift:], src_padding_mask[:, shift:], x[0]
        else:
            encoder_out = x, src_padding_mask

        return encoder_out

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model
