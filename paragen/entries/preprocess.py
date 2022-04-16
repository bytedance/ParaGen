from paragen.entries.util import parse_config
from paragen.tasks import create_task, AbstractTask
from paragen.datasets import create_dataset, AbstractDataset
from paragen.utils.runtime import build_env


def main():
    confs = parse_config()
    if 'env' in confs:
        build_env(confs['task'], **confs['env'])
    task = create_task(confs.pop('task'))
    assert isinstance(task, AbstractTask)
    task.build()
    dataset_conf = confs['dataset']
    for _, conf in confs['data'].items():
        output_path = conf['output_path']
        data_map_path = conf['data_map_path'] if 'data_map_path' in conf else None
        dataset_conf['path'] = conf['path']
        dataset = create_dataset(dataset_conf)
        assert isinstance(dataset, AbstractDataset)
        dataset.build(collate_fn=task._data_collate_fn, preprocessed=False)
        dataset.write(path=output_path, data_map_path=data_map_path)


if __name__ == '__main__':
    main()
