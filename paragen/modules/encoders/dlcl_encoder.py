import torch
import torch.nn as nn
from torch import Tensor

from paragen.modules.encoders import register_encoder
from paragen.modules.layers.dlcl import DynamicLinearCombinationLayer
from paragen.modules.encoders.transformer_encoder import TransformerEncoder


@register_encoder
class DLCLTransformerEncoder(TransformerEncoder):
    """
    TransformerEncoder with Dynamic Linear Combination Layer (DLCL)

    Args:
        
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
                 name=None):
        super().__init__(num_layers=num_layers,
                         d_model=d_model,
                         n_head=n_head,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         attention_dropout=attention_dropout,
                         activation=activation,
                         return_seed=return_seed,
                         learn_pos=learn_pos,
                         normalize_before=normalize_before,
                         embed_scale=embed_scale,
                         embed_layer_norm=embed_layer_norm,
                         name=name)
        self.dlcl_layernorms = None
        self.dlcl_layers = None
        self.dlcl_finallayer = None
    
    def build(self, embed, special_tokens):
        super().build(embed, special_tokens)
        self.dlcl_layernorms = nn.ModuleList([nn.LayerNorm(self._d_model) for i in range(self._num_layers)])
        self.dlcl_layers = nn.ModuleList([DynamicLinearCombinationLayer(idx=i+1, 
                                                                        post_ln=None if self._normalize_before 
                                                                                     else self.dlcl_layernorms[i]) 
                                          for i in range(self._num_layers)])
        self.dlcl_finallayer = DynamicLinearCombinationLayer(idx=self._num_layers+1,
                                                             post_ln=None if self._normalize_before
                                                                          else nn.LayerNorm(self._d_model))
        
    def _forward(self, src: Tensor):
        """
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
        x = x.transpose(0, 1)
        src_padding_mask = src.eq(self._special_tokens['pad'])
        output_stack = [x]

        for idx in range(self._num_layers):
            layer = self._layers[idx]
            dlcl_layer = self.dlcl_layers[idx]
            x = dlcl_layer(torch.stack(output_stack, dim=-1))
            x = layer(x, src_key_padding_mask=src_padding_mask)
            output_stack.append(self.dlcl_layernorms[idx](x) if self._normalize_before else x)
        
        x = self.dlcl_finallayer(torch.stack(output_stack, dim=-1))
        if self._norm is not None:
            x = self._norm(x)
        
        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask
        
        return encoder_out

