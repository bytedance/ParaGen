from typing import Optional

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

from paragen.modules.encoders.layers import AbstractEncoderLayer
from paragen.modules.layers.feed_forward import FFN


class StructuredDropEncoderLayer(AbstractEncoderLayer):
    """
    TransformerEncoderLayer performs one layer of transformer operation, namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        attn_structured_dropout: prob of dropping heads
        ffn_structured_dropout: prob of dropping ffn dim
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0,
                 activation="relu",
                 normalize_before=False,
                 attn_structured_dropout=0.,
                 ffn_structured_dropout=0.,):
        super(StructuredDropEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        self.layerdrop = 0.
        self._dim_feedforward = dim_feedforward
        self._d_model = d_model
        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward=dim_feedforward, activation=activation)
        self._attn_structured_dropout = attn_structured_dropout
        self._ffn_structured_dropout = ffn_structured_dropout

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                :math:`(S, B, D)`, where S is sequence length, B is batch size and D is feature dimension
            src_mask: the attention mask for the src sequence (optional).
                :math:`(S, S)`, where S is sequence length.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, S)`, where B is batch size and S is sequence length
        """
        residual = src
        if self.normalize_before:
            src = self.self_attn_norm(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src)
        if self.training and self.layerdrop > 0.:
            src.div_(1 - self.layerdrop)
        src = residual + src
        if not self.normalize_before:
            src = self.self_attn_norm(src)

        residual = src
        if self.normalize_before:
            src = self.ffn_norm(src)
        src = self.forward_ffn(src)
        src = self.dropout2(src)
        if self.training and self.layerdrop > 0.:
            src.div_(1 - self.layerdrop)
        src = residual + src
        if not self.normalize_before:
            src = self.ffn_norm(src)
        return src
    
    def forward_ffn(self, x: Tensor) -> Tensor:
        if not self.training or self._ffn_structured_dropout < 1e-6:
            x = self.ffn(x)
            return x
        random_probs = np.random.rand(self._dim_feedforward)
        sampled_indeces = np.arange(self._dim_feedforward)[random_probs > self._ffn_structured_dropout]
        x = F.linear(x, self.ffn._fc1.weight[sampled_indeces], self.ffn._fc1.bias[sampled_indeces])
        x.div_(1-self._ffn_structured_dropout)
        x = self.ffn._activation(x)
        x = F.linear(x, self.ffn._fc2.weight[:, sampled_indeces], self.ffn._fc2.bias)
        return x

    

