import json
import argparse
from json import encoder

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention

from paragen.models import AbstractModel, register_model
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.embedding import Embedding
from paragen.modules.utils import get_activation_fn
from paragen.modules.layers.classifier import HuggingfaceClassifier
from paragen.modules.encoders.transformer_encoder import TransformerEncoder
from paragen.modules.utils import param_summary
from paragen.utils.runtime import logger
from paragen.utils.io import UniIO

from .encoderblocks import EncoderBlocks

@register_model
class AbLang(AbstractModel):
    """
    Pretraining model includes Abrep and the head model used for training.
    """
    def __init__(self, hparam, num_labels, path=None, reduce_method='mean'):
        super().__init__(path=path)

        with open(hparam, 'r', encoding='utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))  
        setattr(self.hparams, 'num_labels', num_labels)  

        self._reduce_method = reduce_method
        assert reduce_method in ('mean', 'sum', 'maximum', 'cls', 'tag')

    def _build(self, vocab_size, special_tokens):
        self.AbRep = AbRep(self.hparams)       
        self.AbHead = AbHead(self.hparams)
        numel_train, numel_total = param_summary(self)
        logger.info(f"Summary (trainable/total parameters): {numel_train}M/{numel_total}M")
        
    def forward(self, sequence, germlines, attention_mask=None):
        
        representations = self.AbRep(sequence, attention_mask)
        encoder_output = representations.last_hidden_states

        if self._reduce_method == 'sum':
            decoder_input = encoder_output.sum(dim=1)
        elif self._reduce_method == 'mean':
            decoder_input = encoder_output.mean(dim=1)
        elif self._reduce_method == 'maximum':
            decoder_input = encoder_output.max(dim=1)[0]
        elif self._reduce_method == 'cls':
            decoder_input = encoder_output[:,0,:]
        else:
            decoder_input = encoder_output

        output = self.AbHead(decoder_input)
        
        return output
    
    def get_aa_embeddings(self):
        "This function is used to extract the trained aa_embeddings."
        return self.AbRep.AbEmbeddings.aa_embeddings#().weight.detach()

    def reset(self, mode, *args, **kwargs):
        self._mode = mode

    def load(self, path, device, strict=False):
        """
        Load model from path and move model to device.

        Args:
            path: path to restore model
            device: running device
            strict: load model strictly
        """
        with UniIO(path, 'rb') as fin:
            state_dict = torch.load(fin, map_location=device)
            load_dict = state_dict['model'] if 'model' in state_dict else state_dict

        current_dict = {name: param for name, param in self.named_parameters()}
        check_before = {'loss_keys': [], 'mismatch_keys': []}
        save_load_dict = {}
        for k,v in load_dict.items():
            if k not in current_dict:
                check_before['loss_keys'].append(k)
            elif v.size() != current_dict[k].size():
                check_before['mismatch_keys'].append(k)
            else:
                save_load_dict[k] = v
        logger.info("Check before loading: loss_keys  >>> ")
        if len(check_before['loss_keys']) > 0:
            for ele in check_before['loss_keys']:
                logger.info(f"    - {ele}")
        logger.info("Check before loading: mismatch_keys  >>> ")
        if len(check_before['mismatch_keys']) > 0:
            for ele in check_before['mismatch_keys']:
                load_size, current_size = load_dict[ele].size(), current_dict[ele].size()
                logger.info(f"    - {ele} {load_size} (IN this model {current_size})")
        
        mismatched = self.load_state_dict(save_load_dict, strict=strict)
        if not strict:
            logger.info("keys IN this model but NOT IN loaded model >>> ")
            if len(mismatched.missing_keys) > 0:
                for ele in mismatched.missing_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")
            logger.info("keys NOT IN this model but IN loaded model >>> ")
            if len(mismatched.unexpected_keys) > 0:
                for ele in mismatched.unexpected_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")

    
class AbRep(torch.nn.Module):
    """
    This is the AbRep model.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        self.AbEmbeddings = AbEmbeddings(self.hparams)    
        self.EncoderBlocks = EncoderBlocks(self.hparams)
        
        self.init_weights()
        
    def forward(self, src, attention_mask=None, output_attentions=False):
        
        attention_mask = torch.zeros(*src.shape, device=src.device).masked_fill(src == self.hparams.pad_token_id, 1)

        src = self.AbEmbeddings(src)
        
        output = self.EncoderBlocks(src, attention_mask=attention_mask, output_attentions=output_attentions)
        
        return output
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
    

class AbHead(torch.nn.Module):
    """
    Head for masked sequence prediction.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dense = torch.nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(self.hparams.hidden_size, eps=self.hparams.layer_norm_eps)

        num_labels = self.hparams.num_labels if hasattr(self.hparams, 'num_labels') else self.hparams.vocab_size
        self.decoder = torch.nn.Linear(self.hparams.hidden_size, num_labels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(num_labels))
        
        self.activation = F.gelu
        
        ## self.init_weights() - need to have a function doing this

        self.decoder.bias = self.bias # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`

    def forward(self, features, **kwargs):
        x = self.dense(features)

        x = self.activation(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x



class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.pad_token_id = hparams.pad_token_id
        
        self.AAEmbeddings = torch.nn.Embedding(hparams.vocab_size, hparams.hidden_size, padding_idx=self.pad_token_id)
        self.PositionEmbeddings = torch.nn.Embedding(hparams.max_position_embeddings, hparams.hidden_size, padding_idx=0) # here padding_idx is always 0
        
        self.LayerNorm = torch.nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        self.Dropout = torch.nn.Dropout(hparams.hidden_dropout_prob)

    def forward(self, src):
        
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids = self.create_position_ids_from_input_ids(src, self.pad_token_id)   
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return self.Dropout(self.LayerNorm(embeddings))
        
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(padding_idx).int()
        
        return torch.cumsum(mask, dim=1).long() * mask