import json

from paragen.entries.util import parse_config
from paragen.tasks import create_task, AbstractTask
from paragen.utils.ops import recursive

def main():
    confs = parse_config()
    task = create_task(confs.pop('task'))
    assert isinstance(task, AbstractTask)
    task.build()
    dataloader = task._build_dataloader('train', mode='train')
    output_path = confs['output_path']
    to_list = recursive(lambda x: x.tolist())
    with open(output_path, 'w') as fout:
        for batch in dataloader:
            batch = to_list(batch)
            batch = json.dumps(batch)
            fout.write(f'{batch}\n')


if __name__ == '__main__':
    main()
