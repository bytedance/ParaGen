import math
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import tensorflow as tf

from paragen.generators import AbstractGenerator, register_generator
from paragen.utils.io import remove
from paragen.utils.runtime import Environment


@register_generator
class LightseqTransformerGenerator(AbstractGenerator):
    """
    SequenceGenerator is combination of a model and search algorithm.
    It processes in a multi-step fashion while model processes only one step.
    It is usually separated into encoder and search with decoder, and is
    exported and load with encoder and search module.

    Args:
        path: path to export or load generator
    """

    def __init__(self,
                 batch_size,
                 path=None, ):
        super().__init__(path)
        self._batch_size = batch_size
        env = Environment()
        self._maxlen = getattr(env.configs, 'maxlen', 512)

        self._model = None
        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._lightseq_model = None

    def build_from_model(self, model, src_special_tokens, tgt_special_tokens):
        """
        Build generator from model and search.

        Args:
            model (paragen.models.EncoderDecoder): an encoder-decoder model to be wrapped
            src_special_tokens (dict): source special token dict
            tgt_special_tokens (dict): target special token dict
        """
        self._model = model
        self._src_special_tokens = src_special_tokens
        self._tgt_special_tokens = tgt_special_tokens

    def forward(self, encoder, decoder, search=None):
        """
        Infer a sample as model in evaluation mode.
        Compute encoder output first and decode results with search module

        Args:
            encoder (tuple): encoder inputs
            decoder (tuple): decoder inputs
            search (tuple): search states

        Returns:
            decoder_output: results inferred by search algorithm on decoder
        """
        src = encoder[0].cpu().numpy()
        output, _ = self._lightseq_model.infer(src)
        output = torch.from_numpy(output)
        output = output[:, 0, :]
        return output

    def export(self,
               path,
               net_input,
               lang='en',
               **kwargs):
        """
        Export self to `path` by export model directly

        Args:
            path: path to store serialized model
            net_input: fake net_input for tracing the model
            lang: language
            **kwargs:
                - beam_size: beam search size
                - lenpen: length penalty
                - extra_decode_length: maximum_generation_length = min(src_length + extra_decode_length, max_step)
                - generation_method: generation method
                - topk: top-k candidates
                - topp:
                - diverse_lambda: lambda in diverse
        """
        assert self._model.encoder._normalize_before and self._model.decoder._normalize_before, 'only pre-norm arch can be exported by LightSeq'

        from .transformer_pb2 import Transformer

        transformer = Transformer()

        encoder_state_dict, decoder_state_dict = self._extract_weight()
        self._fill_weight(transformer, encoder_state_dict, decoder_state_dict, lang=lang)

        self._fill_in_conf(transformer, self._model.encoder._n_head, **kwargs)
        self._write(transformer, path)

    def _fill_weight(self, transformer, encoder_state_dict, decoder_state_dict, lang='en'):
        dec_var_name_list = list(decoder_state_dict.keys())
        enc_var_name_list = list(encoder_state_dict.keys())

        # fill each encoder layer's params
        enc_tensor_names = {}
        for name in enc_var_name_list:
            name_split = name.split(".")
            if len(name_split) <= 2 or not name_split[2].isdigit():
                continue
            layer_id = int(name_split[2])
            enc_tensor_names.setdefault(layer_id, []).append(name)

        for layer_id in sorted(enc_tensor_names.keys()):
            fill_layer(
                enc_tensor_names[layer_id],
                encoder_state_dict,
                transformer.encoder_stack.add(),
                enc_layer_mapping_dict,
            )

        # fill each decoder layer's params
        dec_tensor_names = {}
        for name in dec_var_name_list:
            name_split = name.split(".")
            if len(name_split) <= 2 or not name.split(".")[2].isdigit():
                continue
            layer_id = int(name.split(".")[2])
            dec_tensor_names.setdefault(layer_id, []).append(name)

        for layer_id in sorted(dec_tensor_names.keys()):
            fill_layer(
                dec_tensor_names[layer_id],
                decoder_state_dict,
                transformer.decoder_stack.add(),
                dec_layer_mapping_dict,
            )

        # fill src_embedding
        fill_layer(
            enc_var_name_list,
            encoder_state_dict,
            transformer.src_embedding,
            src_emb_mapping_dict,
        )
        src_tb = _gather_token_embedding(
            enc_var_name_list, encoder_state_dict, "_embed"
        )
        transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()

        pos_emb = _get_position_encoding(length=self._maxlen, hidden_size=src_tb.shape[-1])
        pos_emb_list = pos_emb.numpy().reshape([-1]).tolist()
        transformer.src_embedding.position_embedding[:] = pos_emb_list
        logger.info(
            "model.encoder.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
                (pos_emb.shape)
            )
        )

        # fill trg_embedding
        encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
        trg_emb_mapping_dict.update(encode_output_mapping_dict)
        fill_layer(
            dec_var_name_list,
            decoder_state_dict,
            transformer.trg_embedding,
            trg_emb_mapping_dict,
        )
        # assert lang in LANG2ID
        trg_tb = _gather_token_embedding(
            dec_var_name_list, decoder_state_dict, "_embed", lang=lang
        )
        transformer.trg_embedding.token_embedding[:] = trg_tb.transpose().flatten().tolist()
        logger.info(
            "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
                trg_tb.transpose().shape
            )
        )

        pos_emb = _get_position_encoding(length=self._maxlen, hidden_size=trg_tb.shape[-1])
        pos_emb_list = pos_emb.numpy().reshape([-1]).tolist()
        transformer.trg_embedding.position_embedding[:] = pos_emb_list
        logger.info(
            "model.decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
                (pos_emb.shape)
            )
        )

    def _extract_weight(self):
        reloaded = self._model.state_dict()

        encoder_state_dict = {}
        decoder_state_dict = {}
        for k in reloaded:
            if k.startswith("_encoder."):
                encoder_state_dict[k] = reloaded[k]
            if k.startswith("_decoder."):
                decoder_state_dict[k] = reloaded[k]
        decoder_state_dict = split_qkv(decoder_state_dict)
        decoder_state_dict['_decoder.shared_bias'] = decoder_state_dict.pop('_decoder._out_proj_bias')
        return encoder_state_dict, decoder_state_dict

    def _fill_in_conf(self,
                      transformer,
                      nhead,
                      beam_size=4,
                      length_penalty=0.6,
                      extra_decode_length=50,
                      generation_method='beam_search',
                      topk=1,
                      topp=0.75,
                      diverse_lambda=0.,):
        # fill in conf to transformer
        transformer.model_conf.head_num = nhead

        transformer.model_conf.beam_size = beam_size
        transformer.model_conf.length_penalty = length_penalty

        transformer.model_conf.extra_decode_length = extra_decode_length
        transformer.model_conf.src_padding_id = self._src_special_tokens['pad']
        transformer.model_conf.trg_start_id = self._tgt_special_tokens['bos']
        transformer.model_conf.trg_end_id = self._tgt_special_tokens['eos']

        transformer.model_conf.sampling_method = generation_method
        transformer.model_conf.topk = topk
        transformer.model_conf.topp = topp
        transformer.model_conf.diverse_lambda = diverse_lambda
        transformer.model_conf.is_post_ln = False
        transformer.model_conf.no_scale_embedding = False
        transformer.model_conf.use_gelu = False

    def _write(self, transformer, path):
        logger.info("Writing to {0}".format(path))

        try:
            with tf.io.gfile.GFile(path, "wb") as fout:
                fout.write(transformer.SerializeToString())
        except Exception:
            logger.info('Saving PB fails. Save HDF5 instead!')
            remove(path)
            path = path.replace('pb', 'hdf5')
            import h5py
            f = h5py.File(path, "w")
            save_bart_proto_to_hdf5(transformer, f)
            f.close()

    def load(self):
        """
        Load generator from path
        """
        import lightseq.inference as lsi
        self._lightseq_model = lsi.Transformer(self._path, self._batch_size)


""" key是proto参数的值，value是一个强大的表达式，每个&&分割tensor name的匹配路径或表达式，每个匹配
路径的子pattern用空格分隔，表达式用expression_开头，可以对每个tensor进行单独操作，支持多个表达式。多个匹配路径
和表达式最后会concat，axis=-1 """
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "self_attn_norm.weight",
        "multihead_norm_bias": "self_attn_norm.bias",
        "multihead_project_kernel_qkv": "self_attn.in_proj_weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn.in_proj_bias",
        "multihead_project_kernel_output": "self_attn.out_proj.weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn.out_proj.bias",
        "ffn_norm_scale": "ffn_norm.weight",
        "ffn_norm_bias": "ffn_norm.bias",
        "ffn_first_kernel": "ffn._fc1.weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "ffn._fc1.bias",
        "ffn_second_kernel": "ffn._fc2.weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "ffn._fc2.bias",
    }
)

dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "self_attn_norm.weight",
        "self_norm_bias": "self_attn_norm.bias",
        "self_project_kernel_qkv": "self_attn.in_proj_weight&&expression_.transpose(0, 1)",
        "self_project_bias_qkv": "self_attn.in_proj_bias",
        "self_project_kernel_output": "self_attn.out_proj.weight&&expression_.transpose(0, 1)",
        "self_project_bias_output": "self_attn.out_proj.bias",
        "encdec_norm_scale": "multihead_attn_norm.weight",
        "encdec_norm_bias": "multihead_attn_norm.bias",
        "encdec_project_kernel_q": "multihead_attn.q_proj_weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_q": "multihead_attn.q_proj_bias",
        "encdec_project_kernel_output": "multihead_attn.out_proj.weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_output": "multihead_attn.out_proj.bias",
        "ffn_norm_scale": "ffn_norm.weight",
        "ffn_norm_bias": "ffn_norm.bias",
        "ffn_first_kernel": "ffn._fc1.weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "ffn._fc1.bias",
        "ffn_second_kernel": "ffn._fc2.weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "ffn._fc2.bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "_norm.weight",
        "norm_bias": "_norm.bias",
    }
)

trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "_norm.weight",
        "norm_bias": "_norm.bias",
        "shared_bias": "shared_bias",
    }
)


def check_rule(tensor_name, rule):
    if "Adam" in tensor_name or "adam" in tensor_name:
        return False
    assert isinstance(rule, str) and rule
    r_size = len(rule.split('.'))
    t = tensor_name.split('.')
    if len(t) < r_size:
        return False
    return rule == '.'.join(t[-r_size:])


def fill_layer(tensor_names, state_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        expression = [
            ele for ele in ckpt_rule.split("&&") if ele.startswith("expression_")
        ]

        ckpt_rule = [
            ele for ele in ckpt_rule.split("&&") if not ele.startswith("expression_")
        ]

        assert (len(ckpt_rule) > 0 and len(expression) < 2) or (
                len(ckpt_rule) == 0 and len(expression) > 0
        )

        if len(expression) < 2:
            expression = "" if not expression else expression[0].split("_")[1]
        else:
            expression = [exp.split("_")[1] for exp in expression]

        target_tn = []
        for cr in ckpt_rule:
            tmp = []
            for tn in tensor_names:
                if check_rule(tn, cr):
                    tmp.append(tn)
            if len(tmp) != 1:
                logger.info(f'{tmp} {cr}')
            assert len(tmp) == 1
            target_tn.extend(tmp)
        target_tensor = [state_dict[name] for name in target_tn]
        tt = {}
        if target_tensor:
            exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)
        else:
            if not isinstance(expression, list):
                expression = [expression]
            exec("tt['save'] = [%s]" % ",".join(expression))

        target_tensor = np.concatenate(tt["save"], axis=-1)
        logger.info(
            "%s -> %s, shape: %s, convert finished."
            % (target_tn if target_tn else "created", proto_name, target_tensor.shape)
        )
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "{0}.multihead_attn.k_proj_weight&&{0}.multihead_attn.v_proj_weight".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "{0}.multihead_attn.k_proj_bias&&{0}.multihead_attn.v_proj_bias".format(ele)
        for ele in range(dec_layer_num)
    ]

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
    }


def _get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    with tf.device("/cpu:0"):
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = hidden_size // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    return signal


def _gather_token_embedding(tensor_names, name2var_dict, tn_pattern, lang="en"):
    """ use pattern to diff source and target. """
    target_tn = []
    for tn in tensor_names:
        if (tn_pattern in tn.split(".")) and ("weight" in tn.split(".")):
            target_tn.append(tn)
            continue
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    logger.info(
        "token embedding shape is %s, scaled by %s"
        % (target_tensor.shape, target_tensor.shape[1] ** 0.5))

    logger.info("token embedding shape is {}".format(target_tensor.shape))

    return target_tensor


def split_qkv(decoder_state_dict):
    state_dict = OrderedDict()
    for key, val in decoder_state_dict.items():
        if 'multihead_attn.in_proj' in key:
            dim = val.size(0) // 3
            state_dict[key.replace('multihead_attn.in_proj', 'multihead_attn.q_proj')] = val[:dim]
            state_dict[key.replace('multihead_attn.in_proj', 'multihead_attn.k_proj')] = val[dim:dim * 2]
            state_dict[key.replace('multihead_attn.in_proj', 'multihead_attn.v_proj')] = val[dim * 2:]
        else:
            state_dict[key] = val
    return state_dict


def save_bart_proto_to_hdf5(transformer, f):
    """Convert bart protobuf to hdf5 format to support larger weight."""
    MODEL_CONF_KEYS = [
        # model_conf
        "head_num",
        "beam_size",
        "extra_decode_length",
        "length_penalty",
        "src_padding_id",
        "trg_start_id",
        "diverse_lambda",
        "sampling_method",
        "topp",
        "topk",
        "trg_end_id",
        "is_post_ln",
        "no_scale_embedding",
        "use_gelu",
        "is_multilingual",
    ]

    EMBEDDING_KEYS = [
        # src_embedding
        # trg_embedding
        "token_embedding",
        "position_embedding",
        "norm_scale",
        "norm_bias",
        "encode_output_project_kernel_kv",
        "encode_output_project_bias_kv",
        "shared_bias",
        "lang_emb",
        "trg_vocab_mask",
    ]

    ENCODER_LAYER_KEYS = [
        # encoder_stack/{i}
        "multihead_norm_scale",
        "multihead_norm_bias",
        "multihead_project_kernel_qkv",
        "multihead_project_bias_qkv",
        "multihead_project_kernel_output",
        "multihead_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]

    DECODER_LAYER_KEYS = [
        # decoder_stack/{i}
        "self_norm_scale",
        "self_norm_bias",
        "self_project_kernel_qkv",
        "self_project_bias_qkv",
        "self_project_kernel_output",
        "self_project_bias_output",
        "encdec_norm_scale",
        "encdec_norm_bias",
        "encdec_project_kernel_q",
        "encdec_project_bias_q",
        "encdec_project_kernel_output",
        "encdec_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]
    base_attr_to_keys = {
        "src_embedding": EMBEDDING_KEYS,
        "trg_embedding": EMBEDDING_KEYS,
        "model_conf": MODEL_CONF_KEYS,
    }

    from operator import attrgetter

    logger.info(f"start converting protobuf to hdf5 format.")
    # load src_embedding, trg_embedding, model_conf
    for base_attr, keys in base_attr_to_keys.items():
        for key in keys:
            hdf5_key = f"{base_attr}/{key}"
            proto_attr = f"{base_attr}.{key}"

            if key not in dir(attrgetter(base_attr)(transformer)):
                logger.info(f"key {key} not found in {base_attr}, skipping")
                continue

            logger.info(f"loading transformer {proto_attr} -> {hdf5_key}")
            _data = attrgetter(proto_attr)(transformer)
            if type(_data) is str:
                logger.info(
                    f"find type str, explicitly convert string to ascii encoded array."
                )
                # explict convert to array of char (int8) to avoid issues on string reading in C
                _data = np.array([ord(c) for c in _data]).astype(np.int8)
            f.create_dataset(hdf5_key, data=_data)

    # save number of layers metadata
    f.create_dataset("model_conf/n_encoder_stack", data=len(transformer.encoder_stack))
    f.create_dataset("model_conf/n_decoder_stack", data=len(transformer.decoder_stack))

    # load encoder_stack
    for layer_id, layer in enumerate(transformer.encoder_stack):
        for key in ENCODER_LAYER_KEYS:
            hdf5_key = f"encoder_stack/{layer_id}/{key}"
            proto_attr = key
            logger.info(f"loading transformer.encoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    # load decoder_stack
    for layer_id, layer in enumerate(transformer.decoder_stack):
        for key in DECODER_LAYER_KEYS:
            hdf5_key = f"decoder_stack/{layer_id}/{key}"
            proto_attr = key
            logger.info(f"loading transformer.decoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    logger.info(f"proto to hdf5 conversion completed.")
