import os
from paragen.tokenizers import registry, AbstractTokenizer
from paragen.utils.runtime import build_env
from paragen.entries.util import parse_config


def main():
    configs = parse_config()
    if 'env' in configs:
        env_conf = configs.pop('env')
        build_env(configs, **env_conf)
    cls = registry[configs.pop('class').lower()]
    assert issubclass(cls, AbstractTokenizer)
    if '/' in configs['output_path']:
        os.makedirs('/'.join(configs['output_path'].split('/')[:-1]), exist_ok=True)
    data = configs.pop('data')
    cls.learn(data, **configs)


if __name__ == '__main__':
    main()
