import json

from paragen.utils.io import UniIO


def build_txc(config):
    """
    Build a txc structure from config

    Args:
        config: txc configuration path

    Returns:
        - process function
    """
    from txc.modules.structures import create_structure
    from txc.runtime import build_env
    build_env(mode='infer')
    if not config:
        return None
    with UniIO(config) as fin:
        config = json.load(fin)
        txc = create_structure(config)

    def _process(*args):
        for x, unit_in_arg in zip(args, txc.input_names):
            txc.buffer_in[unit_in_arg] = x
        txc.run()
        out = txc.fetch_buffer_out(name=txc.output_names[0])
        return out

    return _process
