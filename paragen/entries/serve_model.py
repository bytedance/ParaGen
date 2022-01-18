from paragen.entries.util import parse_config
import os
import logging
logger = logging.getLogger(__name__)

from thriftpy.rpc import make_server
import thriftpy

from paragen.services.model_server import ModelServer
from paragen.tasks import create_task
from paragen.utils.runtime import build_env


def main():
    configs = parse_config()
    if 'env' in configs:
        build_env(configs['task'], **configs['env'])
    task = create_task(configs.pop('task'))
    task.build()
    generator = task._generator
    model = ModelServer(generator)
    grpc_port = int(os.environ.get('GRPC_PORT', 6000))
    model_infer_thrift = thriftpy.load("/opt/tiger/ParaGen/paragen/services/idls/model_infer.thrift", module_name="model_infer_thrift")
    server = make_server(model_infer_thrift.ModelInfer,
                         model,
                         'localhost',
                         grpc_port,
                         client_timeout=None)
    logger.info('Starting Serving Model')
    server.serve()


if __name__ == '__main__':
    main()
