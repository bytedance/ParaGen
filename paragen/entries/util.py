import argparse
import yaml

from paragen.utils.data import possible_eval
from paragen.utils.io import UniIO


def parse_config():
    """
    Parse configurations from config file and override arguments.
    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='N', type=str, help='config path')
    parser.add_argument('--lib', metavar='N', default=None, type=str, help='customization package')
    parser.add_argument('--local_rank', metavar='N', default=0, type=int, help='local rank when use ddp')
    args, unknown = parser.parse_known_args()
    with UniIO(args.config) as fin:
        confs = yaml.load(fin, Loader=yaml.FullLoader)
    stringizing(confs)
    kv_pairs = []
    current_key = None
    for ele in unknown:
        if ele.startswith("--"):
            current_key = ele[2:]
        else:
            kv_pairs.append((current_key, ele))
    for pair in kv_pairs:
        ks = pair[0].split(".")
        v = possible_eval(pair[1])
        tmp = confs
        last_key = ks[-1]
        for k in ks[:-1]:
            if k not in tmp:
                tmp[k] = {}
            tmp = tmp[k]
        tmp[last_key] = v
    if args.lib:
        if 'env' not in confs:
            confs['env'] = {}
        custom_libs = [args.lib]
        if 'custom_libs' in confs['env']:
            custom_libs.append(confs['env']['custom_libs'])
        confs['env']['custom_libs'] = ','.join(custom_libs)
    if 'env' in confs:
        confs['env']['local_rank'] = args.local_rank
    else:
        confs['env'] = {'local_rank': args.local_rank}
    return confs

def stringizing(conf: dict):
    def _stringizing(def_dct: dict, conf_dct: dict):
        for k, v in conf_dct.items():
            if isinstance(v, str):
                for def_k, def_v in def_dct.items():
                    if def_k in v:
                        v = v.replace(def_k, def_v)
                conf_dct[k] = v
            if isinstance(v, dict):
                _stringizing(def_dct, v)

    if "define" in conf:
        definition = {"${" + k + "}": v for k,v in conf['define'].items()}
        conf.pop("define")
        _stringizing(definition, conf)

