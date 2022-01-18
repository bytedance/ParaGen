import os
import logging
logger = logging.getLogger(__name__)

import euler

from paragen.entries.util import parse_config
from paragen.services import Server, Service
from paragen.utils.runtime import build_env


def main():
    configs = parse_config()

    env = configs.pop('env')
    env['device'] = 'cpu'
    build_env(configs['task'], **env)

    server = Server(configs)
    app = euler.Server(Service)

    @app.register('serve')
    def serve(ctx, req):
        return server.serve(req)

    server_port = int(os.environ.get('SERVER_PORT', 18001))
    logger.info('Starting thrift server in python on PORT {}...'.format(server_port))
    app.run("tcp://0.0.0.0:{}".format(server_port),
            transport="buffered",
            workers_count=getattr(configs, 'worker', 8))
    logger.info('exit!')


if __name__ == '__main__':
    main()
