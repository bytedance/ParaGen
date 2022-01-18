from paragen.tasks import create_task
from paragen.utils.runtime import build_env
from paragen.entries.util import parse_config


def main():
    confs = parse_config()
    if 'env' in confs:
        build_env(confs['task'], **confs['env'])
    task = create_task(confs.pop('task'))
    task.build()
    task.run()


if __name__ == '__main__':
    main()
