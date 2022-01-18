import json
import os
import pickle
import logging
logger = logging.getLogger(__name__)

from thriftpy.rpc import make_client
import thriftpy

from paragen.services import Request, Response
from paragen.tasks import create_task
from paragen.utils.runtime import build_env, Environment


class Server:
    """
    Server make the task a interactive service, and can be deployed online to serve requests.

    Args:
        configs: configurations to build a task
    """

    def __init__(self, configs):
        self._configs = configs

        if 'env' in self._configs:
            build_env(**self._configs['env'])
        task_config = self._configs.pop('task')
        task_config.pop('generator')
        task_config.pop('model')

        self._task = create_task(task_config)
        self._task.build()
        self._task.reset(training=False, infering=True)

        self._env = Environment()

    def serve(self, request):
        """
        Serve a request

        Args:
            request (Request): a request for serving.
                It must contain a jsonable attribute named `samples` indicating a batch of unprocessed samples.

        Returns:
            response (Response): a response to the given request.
        """
        response = Response(results='')
        try:
            logger.info('receive request {}'.format(request))
            generator = _build_backend_generator_service()
            samples = request.samples
            samples = json.loads(samples)
            samples = self._task.preprocess(samples)
            samples = pickle.dumps(samples['net_input'])
            results = generator.infer(samples)
            results = pickle.loads(results)
            debug_info = {'net_output': results.tolist()}
            results = self._task.postprocess(results)
            response = Response(results=json.dumps(results),
                                debug_info=json.dumps(debug_info) if self._env.debug else None)
            logger.info('return response {}'.format(response))
        except Exception as e:
            logger.warning(str(e))
        finally:
            return response


def _build_backend_generator_service():
    """
    Create a service to connect backend neural model

    Returns:
        - a thrift client connecting neural model
    """
    backgronud_model_infer_thrift = thriftpy.load("/opt/tiger/ParaGen/paragen/services/idls/model_infer.thrift",
                                                  module_name="model_infer_thrift")
    grpc_port = int(os.environ.get('GRPC_PORT', 6000))
    generator = make_client(backgronud_model_infer_thrift.ModelInfer,
                            'localhost',
                            grpc_port,
                            timeout=100000000)
    return generator

