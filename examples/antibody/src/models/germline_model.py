import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention

from paragen.models import AbstractModel, register_model
from paragen.models.bert_model import BertModel
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.embedding import Embedding
from paragen.modules.layers.classifier import HuggingfaceClassifier
from paragen.modules.encoders.transformer_encoder import TransformerEncoder
from paragen.modules.utils import param_summary
from paragen.utils.runtime import logger
from paragen.utils.io import UniIO
from paragen.utils.ops import deepcopy_on_ref
from paragen.utils.data import possible_eval


@register_model
class GermlineModel(BertModel):
    def __init__(self,
                 encoder,
                 d_model,
                 fusing_attn_type : str = '-',
                 fusing_num_heads : int = 1, 
                 remove_loaded_head: bool = False,
                 linear_probing : dict = None,
                 path=None):
        super().__init__(encoder, d_model, path=path)
        self._d_model = d_model
        self._fusing_attn_type = fusing_attn_type
        self._fusing_num_heads = fusing_num_heads
        self._remove_loaded_head = remove_loaded_head

        self._linear_probing_configs = linear_probing

        self._max_position = 1024 if 'max_pos' not in self._encoder_config else self._encoder_config['max_pos']

    def _build(self, vocab_size, special_tokens):
        super()._build(vocab_size, special_tokens)

        if self._linear_probing_configs:
            self.linear_probing = self.register_linear_probing()

    def forward(self, sequence, germlines):
        seq, _ = self._encoder(sequence) # (L,BZ,D) (BZ,L)

        germs = []
        for k, germ in germlines.items():
            encode_germ, _ = self._encoder(germ)     # (S,BZ,D)
            germs.append(encode_germ)     
        germs = torch.stack(germs).sum(0)

        germlines = list(germlines.values())

        x = self._fusing_layer(seq, germs)
        x = x.permute(1, 0, 2)

        if self._linear_probing_configs:
            output = self.linear_probing(x)
        else:
            output = x

        return output

    def _fusing_layer(self, encoded_seq, encoded_germ, mutation_mask=None):
        """
            return (L, BZ, D)
        """
        if self._fusing_attn_type == '+':
            x = encoded_seq + encoded_germ
        elif self._fusing_attn_type == '-':
            x = encoded_seq - encoded_germ
        elif self._fusing_attn_type == 'x':
            x = encoded_seq
        elif self._fusing_attn_type == 'g':
            x = encoded_germ
        elif self._fusing_attn_type == 'l2':
            L, BZ, D = encoded_seq.size()
            encoded_seq = encoded_seq.view(-1, D)
            encoded_germ = encoded_germ.view(-1, D)
            w = F.pairwise_distance(encoded_seq, encoded_germ, p=2.0, keepdim=True)
            x = w * encoded_seq
            x = x.view(L, BZ, D)
        elif self._fusing_attn_type == 'cos':
            w = 1 - F.cosine_similarity(encoded_seq, encoded_germ, dim=-1)
            x = w.unsqueeze(-1) * encoded_seq
        else:
            raise NotImplementedError
        return x

    
    def register_linear_probing(self):
        configs = deepcopy_on_ref(self._linear_probing_configs)
        linear_probing_class_name = configs.pop('class')
        if linear_probing_class_name == 'GermlineLabelingLayer':
            linear_probing_class = GermlineLabelingLayer
        elif linear_probing_class_name == 'GermlineClassifierLayer':
            linear_probing_class = GermlineClassifierLayer
        else:
            raise NotImplementedError

        if 'frozen_encoder' in configs.keys():
            frozen_encoder = configs.pop('frozen_encoder')
            if frozen_encoder:
                self._encoder.requires_grad_(False)
                numel_train, numel_total = param_summary(self)
                logger.info(f"Finetune Summary (trainable/total parameters): {numel_train}M/{numel_total}M")

        kwargs = {}
        for k, v in configs.items():
            kwargs[k] = possible_eval(v)

        return linear_probing_class(d_model=self._d_model, **kwargs)

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
            if load_dict['_encoder._embed.weight'].size(0) > self._encoder._embed.weight.size(0):
                logger.info("truncate _encoder._embed.weight from {} to {}".format(load_dict['_encoder._embed.weight'].size(), self._encoder._embed.weight.size(0)))
                load_dict['_encoder._embed.weight'] = load_dict['_encoder._embed.weight'][:self._encoder._embed.weight.size(0),:]
            if self._remove_loaded_head:
                keys = list(load_dict.keys())
                for k in keys:
                    if '_classifier' in k:
                        logger.info("Pop {} from loaded model".format(k))
                        load_dict.pop(k)

            mismatched = self.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict, strict=strict)

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
class EvolutionModel(GermlineModel):
    def __init__(self,
                 encoder,
                 d_model,
                 fusing_attn_type : str = '-',
                 fusing_num_heads : int = 1, 
                 remove_loaded_head: bool = False,
                 linear_probing : dict = None,
                 path=None):
        super().__init__(encoder, d_model, 
                         fusing_attn_type=fusing_attn_type,
                         fusing_num_heads=fusing_num_heads,
                         remove_loaded_head=remove_loaded_head, 
                         linear_probing=linear_probing, 
                         path=path)

        self._max_position = 1024 if 'max_pos' not in self._encoder_config else self._encoder_config['max_pos']

    def _build(self, vocab_size, special_tokens):
        super()._build(vocab_size, special_tokens)
        
        self._special_tokens = special_tokens

    def forward(self, sequence, germlines: dict):
        """
            sequence: Tensor 
                (BZ, L), where BZ is the batch size, L is the source sequence length.
            germlines: Dict 
                {name: Tensor (BZ, L)} * G, where BZ is the batch size, G is the germline number, L is the source sequence length.
        """
        

        germs = list(germlines.values())                # (G, BZ, L)
        sequences = torch.stack([sequence] + germs)     # (S, BZ, L), S = G+1
        sequences = sequences.permute(1, 0, 2)          # (BZ, S, L)

        encoded_seq, encoded_germ = self._get_sequence_repr(sequences)

        x = self._fusing_layer(encoded_seq, encoded_germ)
        x = x.permute(1, 0, 2)                      # (BZ, L, D)
        
        if self._linear_probing_configs:
            output = self.linear_probing(x)
        else:
            output = x

        return output


    def generate(self, sequence, germlines: dict):
        germs = list(germlines.values())                # (G, BZ, L)
        sequences = torch.stack([sequence] + germs)     # (S, BZ, L), S = G+1
        sequences = sequences.permute(1, 0, 2)          # (BZ, S, L)

        encoded_seq, encoded_germ = self._get_sequence_repr(sequences)

        x = self._fusing_layer(encoded_seq, encoded_germ)
        x = x.permute(1, 0, 2) 

        return x

    def _get_sequence_repr(self, sequences):
        batch_size, num_seq, seq_len = sequences.size()
        all_seq, _ = self._encoder(sequences) # (S * L, BZ, D)
        all_seq = all_seq.contiguous().view(num_seq, seq_len, batch_size, -1)
        encoded_seq, encoded_germ = all_seq[0], all_seq[1:]             # (L, BZ, D)
        encoded_germ = encoded_germ.sum(dim=0)                          # (L, BZ, D)
        return encoded_seq, encoded_germ

    def _fusing_layer(self, encoded_seq, encoded_germ, mutation_mask=None):
        return super()._fusing_layer(encoded_seq, encoded_germ, mutation_mask)



@register_model
class EvolutionMutationModel(EvolutionModel):
    def __init__(self,
                encoder,
                d_model,
                fusing_attn_type : str = 'x',
                fusing_num_heads : int = 1, 
                remove_loaded_head: bool = False,
                path=None):
        super().__init__(encoder, d_model, 
                        fusing_attn_type=fusing_attn_type,
                        fusing_num_heads=fusing_num_heads,
                        remove_loaded_head=remove_loaded_head, 
                        linear_probing=None, 
                        path=path)

    def _build(self, vocab_size, special_tokens):
        BertModel._build(self, vocab_size, special_tokens)

        self._special_tokens = special_tokens

        self.position_head = GermlineLabelingLayer(d_model=self._d_model, num_labels=2)
        self.mutation_head = GermlineLabelingLayer(d_model=self._d_model, num_labels=vocab_size)

    def forward(self, sequence, germline, mask):
        """
            sequence: Tensor 
                (BZ, L), where BZ is the batch size, L is the source sequence length.
            germline: Tensor
                {name: Tensor (BZ, L)} * G, where BZ is the batch size, G is the germline number, L is the source sequence length.
        """
        
        # encoded_seq, encoded_germ = self.encodepair(sequence, germline)
        # position_logit = self.position_head(encoded_seq)

        masked_sequence = sequence.masked_fill(mask, self._special_tokens['unk'])
        encoded_masked_seq, encoded_germ = self.encodepair(masked_sequence, germline)

        position_logit = self.position_head(encoded_germ)
        mutation_logit = self.mutation_head(encoded_masked_seq)

        return position_logit, mutation_logit

    def encodepair(self, sequence, germline):
        sequences = torch.stack([sequence, germline])     # (S, BZ, L), S = G+1
        
        sequences = sequences.permute(1, 0, 2)          # (BZ, S, L)
        batch_size, num_seq, seq_len = sequences.size()
        all_seq, _ = self._encoder(sequences) # (S * L, BZ, D)
        all_seq = all_seq.contiguous().view(num_seq, seq_len, batch_size, -1)

        encoded_seq, encoded_germ = all_seq[0], all_seq[1]             # (L, BZ, D)

        return encoded_seq.permute(1, 0, 2), encoded_germ.permute(1, 0, 2)

    def loss(self, sequence, germlines):
        if isinstance(germlines, dict):
            germline = germlines['germline0']
        else:
            germline = germlines

        mask = (sequence != germline)
        position_logit, mutation_logit = self.forward(sequence, germline, mask)

        label = mask.long()
        position_logit = F.log_softmax(position_logit, dim=-1)
        position_logit = position_logit.view(-1, position_logit.size(-1))
        position_loss = F.nll_loss(position_logit, label.view(-1))

        mutation_logit = F.log_softmax(mutation_logit, dim=-1)
        mutation_logit = mutation_logit[label==1]
        mutation_token = sequence[label==1]
        mutation_logit = mutation_logit.view(-1, mutation_logit.size(-1))
        mutation_loss = F.nll_loss(mutation_logit, mutation_token.view(-1))

        ntokens = label.sum()
        position_accu = (position_logit.argmax(dim=-1) == label.view(-1)).float().mean()
        mutation_accu = (mutation_logit.argmax(dim=-1) == mutation_token.view(-1)).float().mean()

        loss = position_loss + mutation_loss

        logging_states = {
            "ntokens":  ntokens.data.item(),            
            "position_loss":    position_loss.data.item(),
            "mutation_loss":    mutation_loss.data.item(),
            "position_accu":    position_accu.data.item(),
            "mutation_accu":    mutation_accu.data.item()
        }

        return loss, logging_states


class GermlineLabelingLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_labels,
                 dropout=0,
                 activation="relu"):
        super().__init__()
        self._d_model = d_model
        self._num_labels = num_labels
        self._classifier = HuggingfaceClassifier(self._d_model, self._num_labels, dropout=dropout, activation=activation)
        
    def forward(self, x):
        logits = self._classifier(x)
        output = logits if self._num_labels > 1 else logits.squeeze(dim=-1)
        return output

class GermlineClassifierLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_labels,
                 reduce_method='cls',
                 dropout=0,
                 activation="relu"):
        super().__init__()
        self._d_model = d_model
        self._num_labels = num_labels
        self._reduce_method = reduce_method
        assert self._reduce_method in ('mean', 'sum', 'maximum', 'cls')

        self._classifier = HuggingfaceClassifier(self._d_model, self._num_labels, dropout=dropout, activation=activation)

    def forward(self, x):
        if self._reduce_method == 'mean':
            x = x.mean(dim=1)
        elif self._reduce_method == 'sum':
            x = x.sum(dim=1)
        elif self._reduce_method == 'maximum':
            x = x.max(dim=1)[0]
        elif self._reduce_method == 'cls':
            x = x[:, 0, :]
        else:
            raise NotImplementedError

        logits = self._classifier(x)
        output = logits if self._num_labels > 1 else logits.squeeze(dim=-1)
        return output
