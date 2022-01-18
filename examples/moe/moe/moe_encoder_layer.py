from typing import Optional

from torch import Tensor
from torch import nn

from paragen.modules.encoders.layers import AbstractEncoderLayer
from .moe import MoE


class MoEEncoderLayer(AbstractEncoderLayer):
    """
    MoEEncoderLayer performs one layer of MoeEncoder.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        num_experts: number of experts in MoE layer. 
            The layer degenerates to standard Transformer layer when num_experts=1.
        sparse: whether using sparse routing (one token only passes one expert).
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0,
                 activation="relu",
                 normalize_before=False,
                 num_experts=1,
                 sparse=True,):
        super(MoEEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        
        self.moe = MoE(d_model, dim_feedforward=dim_feedforward, activation=activation, 
                       num_experts=num_experts, sparse=sparse)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
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
        src = residual + src
        if not self.normalize_before:
            src = self.self_attn_norm(src)

        residual = src
        if self.normalize_before:
            src = self.ffn_norm(src)
        src, loss = self.moe(src)
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.ffn_norm(src)
        return src, loss
