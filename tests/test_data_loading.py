import unittest
import os
import shutil

from paragen.utils.runtime import Environment, build_env


class TestTextDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        from paragen.datasets.text_dataset import TextDataset
        from paragen.samplers.sequential_sampler import SequentialSampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.txt", "w") as f:
            f.write("1 1 1\n")
            f.write("2 2 2 2\n")
            f.write("3 3 3\n")
            f.write("4 4 4 4\n")

        self.dataset = TextDataset("_TMP_/data.txt")
        self.dataset.build()
        self.sampler = SequentialSampler(max_samples=3)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            ["1 1 1", "2 2 2 2", "3 3 3"],
            ["4 4 4 4"]
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        from paragen.datasets.text_dataset import TextDataset
        from paragen.samplers import create_sampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.txt", "w") as f:
            f.write("1 1 1\n")  # should be padded
            f.write("2 2 2\n")  # should be padded
            f.write("3 3 3\n")  # should be padded
            f.write("4 4 4\n")  # should be padded
            f.write("5 5 5\n")
            f.write("6 6 6\n")
            f.write("7 7 7\n")
            f.write("8 8 8\n")

        self.dataset = TextDataset("_TMP_/data.txt")
        self.dataset.build()
        self.sampler = create_sampler({'class': 'SequentialSampler', 'max_samples': 2}, is_training=True)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler)

        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                ['1 1 1', '2 2 2'],
                ['7 7 7', '8 8 8']
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                ["3 3 3", "4 4 4"],
                ["1 1 1", "2 2 2"]
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                ["5 5 5", "6 6 6"],
                ["3 3 3", "4 4 4"]
            ])


class TestJsonDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        import json

        from paragen.datasets.json_dataset import JsonDataset
        from paragen.samplers.sequential_sampler import SequentialSampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.json", "w") as f:
            f.write(f"{json.dumps({'a': '1 1 1 1', 'b': '0 0 0 0'})}\n")
            f.write(f"{json.dumps({'a': '3 3 3 3 3 3', 'b': '2 2 2'})}\n")
            f.write(f"{json.dumps({'a': '5 5', 'b': '4 4 4 4'})}\n")
            f.write(f"{json.dumps({'a': '7 7 7', 'b': '6'})}\n")

        self.dataset = JsonDataset("_TMP_/data.json")
        self.dataset.build()
        self.sampler = SequentialSampler(max_samples=3)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            [{'a': '1 1 1 1', 'b': '0 0 0 0'},
             {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
             {'a': '5 5', 'b': '4 4 4 4'}],
            [{'a': '7 7 7', 'b': '6'}]
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        import json

        from paragen.datasets.json_dataset import JsonDataset
        from paragen.samplers import create_sampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.json", "w") as f:
            f.write(f"{json.dumps({'a': '1 1 1 1', 'b': '0 0 0 0'})}\n")  # should be padded
            f.write(f"{json.dumps({'a': '3 3 3 3 3 3', 'b': '2 2 2'})}\n")  # should be padded
            f.write(f"{json.dumps({'a': '5 5', 'b': '4 4 4 4'})}\n")  # should be padded
            f.write(f"{json.dumps({'a': '7 7 7', 'b': '6'})}\n")  # should be padded
            f.write(f"{json.dumps({'a': '1 1 1', 'b': '0 0 0'})}\n")
            f.write(f"{json.dumps({'a': '3 3 3 3 3', 'b': '2 2'})}\n")
            f.write(f"{json.dumps({'a': '5 5 5', 'b': '4 4 4'})}\n")
            f.write(f"{json.dumps({'a': '7 7', 'b': '6 6 6 6'})}\n")

        self.dataset = JsonDataset("_TMP_/data.json")
        self.dataset.build()
        self.sampler = create_sampler({'class': 'SequentialSampler', 'max_samples': 2}, is_training=True)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler)

        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                [{'a': '1 1 1 1', 'b': '0 0 0 0'}, {'a': '3 3 3 3 3 3', 'b': '2 2 2'}],
                [{'a': '5 5 5', 'b': '4 4 4'}, {'a': '7 7', 'b': '6 6 6 6'}]
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                [{'a': '5 5', 'b': '4 4 4 4'}, {'a': '7 7 7', 'b': '6'}],
                [{'a': '1 1 1 1', 'b': '0 0 0 0'}, {'a': '3 3 3 3 3 3', 'b': '2 2 2'}]
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                [{'a': '1 1 1', 'b': '0 0 0'}, {'a': '3 3 3 3 3', 'b': '2 2'}],
                [{'a': '5 5', 'b': '4 4 4 4'}, {'a': '7 7 7', 'b': '6'}]
            ])


class TestParallelTextDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        from paragen.datasets.parallel_text_dataset import ParallelTextDataset
        from paragen.samplers.sequential_sampler import SequentialSampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.a", "w") as fa, open('_TMP_/data.b', 'w') as fb:
            fa.write('1 1 1 1\n'), fb.write('0 0 0 0\n')
            fa.write('3 3 3 3 3 3\n'), fb.write('2 2 2\n')
            fa.write('5 5\n'), fb.write('4 4 4 4\n')
            fa.write('7 7 7\n'), fb.write('6\n')

        self.dataset = ParallelTextDataset({'a': "_TMP_/data.a", 'b': "_TMP_/data.b"})
        self.dataset.build()
        self.sampler = SequentialSampler(max_samples=3)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            [{'a': '1 1 1 1', 'b': '0 0 0 0'},
             {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
             {'a': '5 5', 'b': '4 4 4 4'}],
            [{'a': '7 7 7', 'b': '6'}]
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        from paragen.datasets.parallel_text_dataset import ParallelTextDataset
        from paragen.samplers import create_sampler
        from paragen.dataloaders.in_memory_dataloader import InMemoryDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.a", "w") as fa, open('_TMP_/data.b', 'w') as fb:
            fa.write('1 1 1 1\n'), fb.write('0 0 0 0\n')  # should be padded
            fa.write('3 3 3 3 3 3\n'), fb.write('2 2 2\n')  # should be padded
            fa.write('5 5\n'), fb.write('4 4 4 4\n')  # should be padded
            fa.write('7 7 7\n'), fb.write('6\n')  # should be padded
            fa.write('1 1 1\n'), fb.write('0 0 0\n')
            fa.write('3 3 3 3 3\n'), fb.write('2 2\n')
            fa.write('5 5 5\n'), fb.write('4 4 4\n')
            fa.write('7 7\n'), fb.write('6 6 6 6\n')

        self.dataset = ParallelTextDataset({'a': "_TMP_/data.a", 'b': "_TMP_/data.b"})
        self.dataset.build()
        self.sampler = create_sampler({'class': 'SequentialSampler', 'max_samples': 2}, is_training=True)
        self.sampler.build(self.dataset)
        self.dataloader = InMemoryDataLoader(self.dataset, self.sampler)

        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                [{'a': '1 1 1 1', 'b': '0 0 0 0'}, {'a': '3 3 3 3 3 3', 'b': '2 2 2'}],
                [{'a': '5 5 5', 'b': '4 4 4'}, {'a': '7 7', 'b': '6 6 6 6'}]
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                [{'a': '5 5', 'b': '4 4 4 4'}, {'a': '7 7 7', 'b': '6'}],
                [{'a': '1 1 1 1', 'b': '0 0 0 0'}, {'a': '3 3 3 3 3 3', 'b': '2 2 2'}]
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                [{'a': '1 1 1', 'b': '0 0 0'}, {'a': '3 3 3 3 3', 'b': '2 2'}],
                [{'a': '5 5', 'b': '4 4 4 4'}, {'a': '7 7 7', 'b': '6'}]
            ])


class TestStreamingTextDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        from paragen.datasets.streaming_text_dataset import StreamingTextDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.txt", "w") as f:
            f.write("1 1 1\n")
            f.write("2 2 2 2\n")
            f.write("3 3 3\n")
            f.write("4 4 4 4\n")

        self.dataset = StreamingTextDataset("_TMP_/data.txt")
        self.dataset.build()
        self.dataloader = StreamingDataLoader(self.dataset, length_interval=1, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            "1 1 1",
            "2 2 2 2",
            "3 3 3",
            "4 4 4 4"
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        from paragen.datasets.streaming_text_dataset import StreamingTextDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.txt", "w") as f:
            f.write("1 1 1\n")
            f.write("2 2 2\n")
            f.write("3 3 3\n")
            f.write("4 4 4\n")
            f.write("5 5 5\n")
            f.write("6 6 6\n")
            f.write("7 7 7\n")  # should be dropped
            f.write("8 8 8\n")  # should be dropped

        self.dataset = StreamingTextDataset("_TMP_/data.txt")
        self.dataset.build()
        self.dataset.reset()
        self.dataloader = StreamingDataLoader(self.dataset)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                "1 1 1",
                "4 4 4",
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                "2 2 2",
                "5 5 5",
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                "3 3 3",
                "6 6 6",
            ])


class TestStreamingJsonDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        import json

        from paragen.datasets.streaming_json_dataset import StreamingJsonDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.json", "w") as f:
            f.write(f"{json.dumps({'a': '1 1 1 1', 'b': '0 0 0 0'})}\n")
            f.write(f"{json.dumps({'a': '3 3 3 3 3 3', 'b': '2 2 2'})}\n")
            f.write(f"{json.dumps({'a': '5 5', 'b': '4 4 4 4'})}\n")
            f.write(f"{json.dumps({'a': '7 7 7', 'b': '6'})}\n")

        self.dataset = StreamingJsonDataset("_TMP_/data.json")
        self.dataset.build()
        self.dataloader = StreamingDataLoader(self.dataset, length_interval=1, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            {'a': '1 1 1 1', 'b': '0 0 0 0'},
            {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
            {'a': '5 5', 'b': '4 4 4 4'},
            {'a': '7 7 7', 'b': '6'}
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        import json

        from paragen.datasets.streaming_json_dataset import StreamingJsonDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.json", "w") as f:
            f.write(f"{json.dumps({'a': '1 1 1 1', 'b': '0 0 0 0'})}\n")
            f.write(f"{json.dumps({'a': '3 3 3 3 3 3', 'b': '2 2 2'})}\n")
            f.write(f"{json.dumps({'a': '5 5', 'b': '4 4 4 4'})}\n")
            f.write(f"{json.dumps({'a': '7 7 7', 'b': '6'})}\n")
            f.write(f"{json.dumps({'a': '1 1 1', 'b': '0 0 0'})}\n")
            f.write(f"{json.dumps({'a': '3 3 3 3 3', 'b': '2 2'})}\n")
            f.write(f"{json.dumps({'a': '5 5 5', 'b': '4 4 4'})}\n") # should be dropped
            f.write(f"{json.dumps({'a': '7 7', 'b': '6 6 6 6'})}\n")  # should be dropped

        self.dataset = StreamingJsonDataset("_TMP_/data.json")
        self.dataset.build()
        self.dataset.reset()
        self.dataloader = StreamingDataLoader(self.dataset)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                {'a': '1 1 1 1', 'b': '0 0 0 0'},
                {'a': '7 7 7', 'b': '6'},
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
                {'a': '1 1 1', 'b': '0 0 0'},
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                {'a': '5 5', 'b': '4 4 4 4'},
                {'a': '3 3 3 3 3', 'b': '2 2'},
            ])


class TestStreamingParallelTextDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs("_TMP_", exist_ok=True)
        build_env()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("_TMP_")

    def test_iterator_num_workers(self):
        for i in range(2):
            self._test_iterator_num_workers(num_workers=i)

    def _test_iterator_num_workers(self, num_workers):

        from paragen.datasets.streaming_parallel_text_dataset import StreamingParallelTextDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = 0
        env.rank = 0
        env.distributed_world = 1

        with open("_TMP_/data.a", "w") as fa, open('_TMP_/data.b', 'w') as fb:
            fa.write('1 1 1 1\n'), fb.write('0 0 0 0\n')
            fa.write('3 3 3 3 3 3\n'), fb.write('2 2 2\n')
            fa.write('5 5\n'), fb.write('4 4 4 4\n')
            fa.write('7 7 7\n'), fb.write('6\n')

        self.dataset = StreamingParallelTextDataset({'a': "_TMP_/data.a", 'b': "_TMP_/data.b"})
        self.dataset.build()
        self.dataloader = StreamingDataLoader(self.dataset, length_interval=1, num_workers=num_workers)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        self.assertListEqual(lines, [
            {'a': '1 1 1 1', 'b': '0 0 0 0'},
            {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
            {'a': '5 5', 'b': '4 4 4 4'},
            {'a': '7 7 7', 'b': '6'}
        ])

    def test_iterator_multiprocessing(self):
        for rank in range(3):
            self._test_iterator_multiprocessing(rank)

    def _test_iterator_multiprocessing(self, rank):

        from paragen.datasets.streaming_parallel_text_dataset import StreamingParallelTextDataset
        from paragen.dataloaders.streaming_dataloader import StreamingDataLoader

        env = Environment()
        env.local_rank = rank
        env.rank = rank
        env.distributed_world = 3

        with open("_TMP_/data.a", "w") as fa, open('_TMP_/data.b', 'w') as fb:
            fa.write('1 1 1 1\n'), fb.write('0 0 0 0\n')
            fa.write('3 3 3 3 3 3\n'), fb.write('2 2 2\n')
            fa.write('5 5\n'), fb.write('4 4 4 4\n')
            fa.write('7 7 7\n'), fb.write('6\n')
            fa.write('1 1 1\n'), fb.write('0 0 0\n')
            fa.write('3 3 3 3 3\n'), fb.write('2 2\n')
            fa.write('5 5 5\n'), fb.write('4 4 4\n')  # should be dropped
            fa.write('7 7\n'), fb.write('6 6 6 6\n')  # should be dropped

        self.dataset = StreamingParallelTextDataset({'a': "_TMP_/data.a", 'b': "_TMP_/data.b"})
        self.dataset.build()
        self.dataset.reset()
        self.dataloader = StreamingDataLoader(self.dataset)
        lines = []
        for line in self.dataloader:
            lines.append(line)
        if rank == 0:
            self.assertListEqual(lines, [
                {'a': '1 1 1 1', 'b': '0 0 0 0'},
                {'a': '7 7 7', 'b': '6'},
            ])
        elif rank == 1:
            self.assertListEqual(lines, [
                {'a': '3 3 3 3 3 3', 'b': '2 2 2'},
                {'a': '1 1 1', 'b': '0 0 0'},
            ])
        elif rank == 2:
            self.assertListEqual(lines, [
                {'a': '5 5', 'b': '4 4 4 4'},
                {'a': '3 3 3 3 3', 'b': '2 2'},
            ])


if __name__ == '__main__':
    unittest.main()
