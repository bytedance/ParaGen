from typing import Optional

from torch import Tensor

from paragen.modules.encoders.layers import AbstractEncoderLayer
from paragen.utils.runtime import Environment


class LSTransformerEncoderLayer(AbstractEncoderLayer):
    """
    TransformerEncoderLayer performs one layer of transformer operation, namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0,
                 activation='relu',
                 activation_dropout=0.,
                 normalize_before=False,
                 max_tokens=4096,
                 maxlen=256):
        super(LSTransformerEncoderLayer, self).__init__()
        env = Environment()
        from lightseq.training.ops.pytorch.transformer_encoder_layer import LSTransformerEncoderLayer as TransformerEncoderLayer
        config = TransformerEncoderLayer.get_config(
            max_batch_tokens=max_tokens,
            max_seq_len=maxlen,
            hidden_size=d_model,
            intermediate_size=dim_feedforward,
            nhead=nhead,
            attn_prob_dropout_ratio=attention_dropout,
            activation_fn=activation,
            activation_dropout_ratio=activation_dropout,
            hidden_dropout_ratio=dropout,
            pre_layer_norm=normalize_before,
            fp16=env.fp16,
            local_rank=env.local_rank,
        )
        self._layer = TransformerEncoderLayer(config)

    def forward(self,
                src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                :math:`(S, B, D)`, where S is sequence length, B is batch size and D is feature dimension
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, S)`, where B is batch size and S is sequence length
        """
        return self._layer(src, src_key_padding_mask)
