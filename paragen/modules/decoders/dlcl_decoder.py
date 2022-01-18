import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

from paragen.modules.decoders import register_decoder
from paragen.modules.layers.dlcl import DynamicLinearCombinationLayer
from paragen.modules.decoders.nonauto_transformer_decoder import NonAutoTransformerDecoder


@register_decoder
class DLCLNonAutoTransformerDecoder(NonAutoTransformerDecoder):
    """
    NonAutoTransformerDecoder with Dynamic Linear Combination Layer (DLCL)

    Args:

    """
    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 learn_pos=False,
                 normalize_before=False,
                 output_bias=False,
                 embed_layer_norm=False,
                 name=None):
        super().__init__(num_layers=num_layers,
                         d_model=d_model,
                         n_head=n_head,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         attention_dropout=attention_dropout,
                         activation=activation,
                         learn_pos=learn_pos,
                         normalize_before=normalize_before,
                         output_bias=output_bias,
                         embed_layer_norm=embed_layer_norm,
                         name=name)
        self.dlcl_layernorms = None
        self.dlcl_layers = None
        self.dlcl_finallayer = None

    def build(self, embed, special_tokens, out_proj):
        super().build(embed, special_tokens, out_proj)
        self.dlcl_layernorms = nn.ModuleList([nn.LayerNorm(self._d_model) for i in range(self._num_layers)])
        self.dlcl_layers = nn.ModuleList([DynamicLinearCombinationLayer(idx=i+1, 
                                                                        post_ln=None if self._normalize_before 
                                                                                     else self.dlcl_layernorms[i]) 
                                          for i in range(self._num_layers)])
        self.dlcl_finallayer = DynamicLinearCombinationLayer(idx=self._num_layers+1,
                                                             post_ln=None if self._normalize_before
                                                                          else nn.LayerNorm(self._d_model))
    
    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_padding_mask,
                memory_padding_mask: Optional[torch.Tensor] = None) -> Tensor:
        x = tgt
        if self._pos_embed is not None:
            pos_embed = self._pos_embed(tgt_padding_mask.long())
            x = x + pos_embed
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)
        x = x.transpose(0, 1)
        output_stack = [x]

        for idx in range(self._num_layers):
            layer = self._layers[idx]
            dlcl_layer = self.dlcl_layers[idx]
            x = dlcl_layer(torch.stack(output_stack, dim=-1))
            x = layer(tgt=x,
                      memory=memory,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask)
            output_stack.append(self.dlcl_layernorms[idx](x) if self._normalize_before else x)
        
        x = self.dlcl_finallayer(torch.stack(output_stack, dim=-1))
        if self._norm is not None:
            x = self._norm(x)
        x = x.transpose(0, 1)

        logits = self._out_proj(x)
        return logits
