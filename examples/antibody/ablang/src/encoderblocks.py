import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.multihead_attention import MultiheadAttention


@dataclass
class AbRepOutput():
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

        
class EncoderBlocks(torch.nn.Module):
    """
    Wrapper for multiple EncoderBlocks (or a single).
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.Layers = nn.ModuleList([EncoderBlock(hparams) for _ in range(hparams.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False):
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for num_block, a_EncoderBlock in enumerate(self.Layers):
            
            hidden_states, attentions = a_EncoderBlock(hidden_states, attention_mask, output_attentions)
            #print(attentions)
            
            if output_hidden_states: 
                all_hidden_states = all_hidden_states + (hidden_states,) # Takes out each hidden states after each EncoderBlock
            
            if output_attentions: 
                all_self_attentions = all_self_attentions + (attentions,) # Takes out attention layers for analysis
                  
        return AbRepOutput(last_hidden_states=hidden_states, all_hidden_states=all_hidden_states, attentions=all_self_attentions)
    
    
class EncoderBlock(torch.nn.Module):
    """
    Single EncoderBlock.
    
    An EncoderBlock consists of a MultiHeadAttention and a IntermediateLayer.
    """
    def __init__(self, hparams):
        super().__init__()
        
        self.MultiHeadAttention = ThirdMultiHeadAttention(hparams)
        self.MHADropout = nn.Dropout(hparams.hidden_dropout_prob)
        self.MHALayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        
        self.IntermediateLayer = IntermediateLayer(hparams)
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):

        MHAoutput, attentions = self.MultiHeadAttention(hidden_states, attention_mask, output_attentions=output_attentions)
         
        output = self.MHADropout(MHAoutput)
        output = self.MHALayerNorm(output + hidden_states) # HIDDEN_STATES ARE ADDED FOR RESIDUAL BLOCK EFFECT
        
        output = self.IntermediateLayer(output) # INTERMEDIATELAYER HAS RESIDUAL BLOCK EFFECT INTERNALLY

        #outputs = (layer_output,) + self_attention_outputs[1:]  # if output_attentions=False then 1: is empty
        
        return output, attentions
    
    
class ThirdMultiHeadAttention(torch.nn.Module):
    """
    New MultiHeadAttention which can return the weights of the individual heads.
    """
    
    def __init__(self, hparams):
        super().__init__()
                
        self.Attention = MultiheadAttention(hparams.hidden_size, hparams.num_attention_heads, dropout=hparams.attention_probs_dropout_prob, self_attention=True)
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
                
        hidden_states = torch.transpose(hidden_states, 0, 1)
        
        # static_kv is only True because there is currently a bug which doesn't return the head weights unaveraged unless its true
        attn_output, attn_weights = self.Attention(hidden_states, hidden_states, hidden_states, key_padding_mask=attention_mask, static_kv=True, 
                                                   need_weights=output_attentions, need_head_weights=output_attentions)

        return torch.transpose(attn_output, 0, 1), attn_weights


class OldMultiHeadAttention(torch.nn.Module):
    """
    MultiHeadAttention contains a Scaled Dot Product Attention and a Linear Layer.    
    """
    def __init__(self, config):
        super().__init__()
        self.Attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):  
        
        hidden_states = torch.transpose(hidden_states, 0, 1)
        output, attentions = self.Attention(hidden_states, hidden_states, hidden_states, key_padding_mask=attention_mask, need_weights=output_attentions) 
        
        attention_output = torch.transpose(output, 0, 1)

        return attention_output, attentions
    

class IntermediateLayer(nn.Module):
    """
    Contains an expanding layer, while also functioning as a residual block ending with a drop-norm layer
    """
    def __init__(self, config):
        super().__init__()
        self.expand_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu
        
        self.dense_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        output = self.expand_dense(hidden_states)
        output = self.intermediate_act_fn(output)
               
        output = self.dense_dense(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)
                
        return output 
