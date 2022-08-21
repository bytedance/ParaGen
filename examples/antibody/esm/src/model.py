import json
import argparse
from json import encoder
from pickle import load

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

from .esm import pretrained


class BertLMHead(torch.nn.Module):

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = torch.nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        if weight is None:
            weight = torch.nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        # Only project the masked tokens while training, saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

@register_model
class ESMLM(AbstractModel):
    def __init__(self, model_name, d_model, layer_num, path=None):
        super().__init__(path=path)

        self._model_name = model_name
        self._d_model = d_model
        self._layer_num = layer_num

    def _build(self, vocab_size, special_tokens):
        self._encoder, _ = pretrained.load_model_and_alphabet(self._model_name)
        # pretrained_load = getattr(pretrained, self._model_name)
        # self._encoder, _ = pretrained_load()

        self._lm_head = BertLMHead(
            embed_dim=self._d_model, 
            output_dim=vocab_size, 
            activation_fn='gelu',
            weight=self._encoder.embed_tokens.weight
        )

        numel_train, numel_total = param_summary(self)
        logger.info(f"Summary (trainable/total parameters): {numel_train}M/{numel_total}M")
        
    def forward(self, src_tokens, mask, **kwargs):
        encoder_output = self._encoder(src_tokens, repr_layers=[self._layer_num], return_contacts=False)
        encoder_output = encoder_output["representations"][self._layer_num]

        scores = self._lm_head(encoder_output, mask)
        logits = F.log_softmax(scores, dim=-1)
        return logits

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

    def loss(self, src_tokens, mask, tgt_tokens):
        logits = self.forward(src_tokens, mask)

        target = tgt_tokens[mask]
        ntokens = mask.sum()

        accu = (logits.argmax(1) == target).float().mean()
        loss = F.nll_loss(logits, target)

        logging_states = {
            "loss":     loss.data.item(),
            "ntokens":  ntokens.data.item(),
            "accu":     accu.data.item()
        }

        return loss, logging_states

@register_model
class ESMPredictor(AbstractModel):
    def __init__(self, model_name, d_model, layer_num, num_labels, path=None, reduce_method='mean'):
        super().__init__(path=path)

        self._model_name = model_name
        self._d_model = d_model
        self._layer_num = layer_num
        self._num_labels = num_labels
        self._classifier = HuggingfaceClassifier(self._d_model, self._num_labels)

        self._reduce_method = reduce_method
        assert reduce_method in ('mean', 'sum', 'maximum', 'cls', 'tag')

    def _build(self, vocab_size, special_tokens):
        self._encoder, _ = pretrained.load_model_and_alphabet(self._model_name)
        # pretrained_load = getattr(pretrained, self._model_name)
        # self._encoder, _ = pretrained_load()

        numel_train, numel_total = param_summary(self)
        logger.info(f"Summary (trainable/total parameters): {numel_train}M/{numel_total}M")
        
    def forward(self, sequence, germlines):
        encoder_output = self._encoder(sequence, repr_layers=[self._layer_num], return_contacts=False)
        encoder_output = encoder_output["representations"][self._layer_num]

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

        logits = self._classifier(decoder_input)
        output = logits if self._num_labels > 1 else logits.squeeze(dim=-1)
        return output

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

@register_model
class ESMMSAPredictor(ESMPredictor):
    def __init__(self, model_name, d_model, layer_num, num_labels, path=None, reduce_method='mean'):
        super().__init__(model_name, d_model, layer_num, num_labels, path, reduce_method=reduce_method)

    def forward(self, sequence, germlines):
        germs = list(germlines.values())                # (G, BZ, L)
        sequences = torch.stack([sequence] + germs)     # (S, BZ, L), S = G+1
        sequences = sequences.permute(1, 0, 2)          # (BZ, S, L)
        batch_size, num_seq, seq_len = sequences.size()

        encoder_output = self._encoder(sequences, repr_layers=[self._layer_num], return_contacts=False)
        all_seq = encoder_output["representations"][self._layer_num]    # (BZ, S, L, D)
        all_seq = all_seq.contiguous().permute(1, 0, 2, 3)              # (S, BZ, L, D)

        encoder_output, encoded_germ = all_seq[0], all_seq[1:]             # (BZ, L, D)

        if self._reduce_method == 'sum':
            decoder_input = encoder_output[:,1:,:].sum(dim=1)
        elif self._reduce_method == 'mean':
            decoder_input = encoder_output[:,1:,:].mean(dim=1)
        elif self._reduce_method == 'maximum':
            decoder_input = encoder_output[:,1:,:].max(dim=1)[0]
        elif self._reduce_method == 'cls':
            decoder_input = encoder_output[:,0,:]
        else:
            decoder_input = encoder_output

        logits = self._classifier(decoder_input)
        output = logits if self._num_labels > 1 else logits.squeeze(dim=-1)
        return output
