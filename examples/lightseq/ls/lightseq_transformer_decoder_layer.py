from typing import Dict, Optional

from torch import Tensor
import torch

from paragen.modules.decoders.layers import AbstractDecoderLayer
from paragen.utils.runtime import Environment


class LSTransformerDecoderLayer(AbstractDecoderLayer):
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
                 activation='relu',
                 dropout=0.1,
                 attention_dropout=0,
                 activation_dropout=0.,
                 normalize_before=False,
                 max_tokens=4096,
                 maxlen=256):
        super(LSTransformerDecoderLayer, self).__init__()
        env = Environment()

        from lightseq.training.ops.pytorch.transformer_decoder_layer import LSTransformerDecoderLayer as TransformerDecoderLayer
        config = TransformerDecoderLayer.get_config(
            max_batch_tokens=max_tokens,
            max_seq_len=maxlen,
            hidden_size=d_model,
            intermediate_size=dim_feedforward,
            nhead=nhead,
            activation_fn=activation,
            attn_prob_dropout_ratio=attention_dropout,
            activation_dropout_ratio=activation_dropout,
            hidden_dropout_ratio=dropout,
            pre_layer_norm=normalize_before,
            fp16=env.fp16,
            local_rank=env.local_rank,
            nlayer=env.configs['model']['decoder']['num_layers']
        )
        self._layer = TransformerDecoderLayer(config)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Pass the inputs (and mask) through the decoder layer in training mode.

        Args:
            tgt: the sequence to the decoder layer (required).
                :math:`(B, T, D)`, where T is sequence length, B is batch size and D is feature dimension
            memory: the sequence from the last layer of the encoder (required).
                :math:`(M, B, D)`, where M is memory size, B is batch size and D is feature dimension
            tgt_mask: the mask for the tgt sequence (optional).
                :math:`(T, T)`, where T is sequence length.
            memory_mask: the mask for the memory sequence (optional).
                :math:`(M, M)`, where M is memory size.
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                :math: `(B, T)`, where B is batch size and T is sequence length.
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
                :math: `(B, M)`, where B is batch size and M is memory size.
        """
        if self._mode == 'infer':
            if 'memory' in self._cache:
                memory = self._cache['memory']
            if 'memory_key_padding_mask' in self._cache:
                memory_key_padding_mask = self._cache['memory_key_padding_mask']
            tgt = tgt[:, -1:]
        output = self._layer(tgt, memory, memory_key_padding_mask, self._cache if self._mode == 'infer' else None)
        if self._mode == 'infer':
            if 'memory' not in self._cache:
                self._cache['memory'] = memory
            if 'memory_key_padding_mask' not in self._cache:
                self._cache['memory_key_padding_mask'] = memory_key_padding_mask
        return output

    def reset(self, mode: str):
        """
        Reset encoder layer and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache: Dict[str, torch.Tensor] = {}
        self._mode = mode

    def get_cache(self):
        _, nhead, seqlen, dim = self._cache['dec_self_k'].size()
        cache = {
            'dec_self_k': self._cache['dec_self_k'].transpose(1, 2).transpose(0, 1).reshape(seqlen, -1, nhead * dim),
            'dec_self_v': self._cache['dec_self_v'].transpose(1, 2).transpose(0, 1).reshape(seqlen, -1, nhead * dim),
        }
        return cache

    def set_cache(self, cache: Dict[str, torch.Tensor]):
        _, nhead, seqlen, dim = self._cache['dec_self_k'].size()
        self._cache['dec_self_k'] = cache['dec_self_k'].reshape(seqlen, -1, nhead, dim).transpose(0, 1).transpose(1, 2)
        self._cache['dec_self_v'] = cache['dec_self_v'].reshape(seqlen, -1, nhead, dim).transpose(0, 1).transpose(1, 2)
