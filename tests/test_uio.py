import os
import sys
import unittest
import torch
import numpy
import pickle
import time
from paragen.utils.io import UniIO
from paragen.models import create_model

LOCAL_PATH = '<local path>'
HDFS_PATH = '<hdfs path>'
DELAY=2

class TestUio(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    def test_local_text_io(self):
        with self.subTest(msg='test local single file text io'):
            self._test_text_io_single_file(LOCAL_PATH)
        with self.subTest(msg='test local multi file text io'):
            self._test_text_io_multi_file(LOCAL_PATH)

    def test_hdfs_text_io(self):
        with self.subTest(msg='test hdfs single file text io'):
            self._test_text_io_single_file(HDFS_PATH)
        with self.subTest(msg='test hdfs multi file text io'):
            self._test_text_io_multi_file(HDFS_PATH)

    def test_local_bytes_io(self):
        with self.subTest(msg='test local single file bytes io'):
            self._test_byte_io(LOCAL_PATH)
        with self.subTest(msg='test local multi file bytes io'):
            self._test_byte_io(LOCAL_PATH, method='multi')

    def test_hdfs_bytes_io(self):
        with self.subTest(msg='test hdfs single file bytes io'):
            self._test_byte_io(HDFS_PATH)
        with self.subTest(msg='test hdfs multi file bytes io'):
            self._test_byte_io(HDFS_PATH, method='multi')

    def test_pickle(self):
        with self.subTest(msg='test pickle locally'):
            self._test_pickle(LOCAL_PATH)
        with self.subTest(msg='test pickle in hdfs'):
            self._test_pickle(HDFS_PATH)

    def test_torch_save(self):
        with self.subTest(msg='test torch save locally'):
            self._test_torch_save(LOCAL_PATH)
        with self.subTest(msg='test torch save in hdfs'):
            self._test_torch_save(HDFS_PATH)

    def _test_text_io_single_file(self, pathprefix):
        content = [
            '1 2 3 4 5\n', 
            '6 7 8 9 10\n',
            'Hello world\n',
            'ByteDance\n',
            '\t      \t\n',
            '\n',
            'å∫ç∂´ƒ©',
        ]
        f = UniIO(os.path.join(pathprefix, 'unit_test.text.single.1'), 'w', encoding='utf8')
        f.write(''.join(content))
        f.close()
        time.sleep(DELAY)
        with self.subTest(msg='test readline'):
            f = UniIO(os.path.join(pathprefix, 'unit_test.text.single.1'), 'r', encoding='utf8')
            idx = 0
            while True:
                line = f.readline()
                if line:
                    self.assertEqual(line, content[idx])
                    idx += 1
                else:
                    break
            line = f.readline()
            self.assertEqual(line, '')
            f.close()
        with self.subTest(msg='test readlines'):
            f = UniIO(os.path.join(pathprefix, 'unit_test.text.single.1'), 'r', encoding='utf8')
            lines = f.readlines()
            self.assertListEqual(lines, content)
            lines = f.readlines()
            self.assertListEqual(lines, [])
            f.close()
        with self.subTest(msg='test iteration'):
            f = UniIO(os.path.join(pathprefix, 'unit_test.text.single.1'), 'r', encoding='utf8')
            idx = 0
            for line in f:
                self.assertEqual(line, content[idx])
                idx += 1
            with self.assertRaises(StopIteration):
                line = next(f)
            f.close()
        with self.subTest(msg='test context manager'):
            with UniIO(os.path.join(pathprefix, 'unit_test.text.single.1'), 'r', encoding='utf8') as f:
                lines = f.readlines()
            self.assertListEqual(lines, content)
                
    def _test_text_io_multi_file(self, pathprefix):
        contents = [
            [
                'Hello world\n',
                'ByteDance\n',
                'Inspire Creativity Enrich Life',
            ],
            [
                '你好，世界\n',
                '字节跳动\n',
                '激发创意，丰富生活',
            ],
            [
                'Salve mundi\n',
                'ByteDance\n',
                'Inspíra Enrich Lorem vitae',
            ],
        ]
        for i, content in enumerate(contents):
            with UniIO(os.path.join(pathprefix, f'unit_test.text.multi.{i}'), 'w') as f:
                f.write(''.join(content))
        time.sleep(DELAY)
        contents = [item for content in contents for item in content]
        with self.subTest(msg='test iteration'):
            idx = 0
            with UniIO(os.path.join(pathprefix, 'unit_test.text.multi*'), 'r') as f:
                for line in f:
                    self.assertEqual(line, contents[idx])
                    idx += 1
                with self.assertRaises(StopIteration):
                    line = next(f)
        with self.subTest(msg='test readlines'):
            with UniIO(os.path.join(pathprefix, 'unit_test.text.multi*'), 'r') as f:
                lines = f.readlines()
                self.assertListEqual(lines, contents)
                lines = f.readlines()
                self.assertListEqual(lines, [])
        with self.subTest(msg='test readline'):
            idx = 0
            with UniIO(os.path.join(pathprefix, 'unit_test.text.multi*'), 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        self.assertEqual(line, contents[idx])
                        idx += 1
                    else:
                        break
                line = f.readline()
                self.assertEqual(line, '')

    def _test_byte_io(self, pathprefix, method='single'):
        content1 = os.urandom(65536).replace(b'\n', b'\x00')
        content2 = os.urandom(65536).replace(b'\n', b'\x00')
        content3 = os.urandom(65536).replace(b'\n', b'\x00')
        filename = 'unit_test.bytes.single' if method == 'single' else 'unit_test.bytes.multi'
        if method == 'single':
            with UniIO(os.path.join(pathprefix, filename), 'wb') as f:
                f.write(content1)
                f.write(b'\n')
                f.write(content2)
                f.write(b'\n')
                f.write(content3)
        else:
            with UniIO(os.path.join(pathprefix, filename + '.1'), 'wb') as f:
                f.write(content1)
                f.write(b'\n')
            with UniIO(os.path.join(pathprefix, filename + '.2'), 'wb') as f:
                f.write(content2)
                f.write(b'\n')
            with UniIO(os.path.join(pathprefix, filename + '.3'), 'wb') as f:
                f.write(content3)
        time.sleep(DELAY)
        with self.subTest(msg='test readline'):
            with UniIO(os.path.join(pathprefix, filename + '*'), 'rb') as f:
                line = f.readline()
                self.assertEqual(line, content1 + b'\n')
                line = f.readline(65536)
                self.assertEqual(line, content2)
                line = f.readline(1)
                self.assertEqual(line, b'\n')
                line = f.readline()
                self.assertEqual(line, content3)
        with self.subTest(msg='test read'):
            with UniIO(os.path.join(pathprefix, filename + '*'), 'rb') as f:
                readbytes = f.read()
                self.assertEqual(readbytes, content1 + b'\n' + content2 + b'\n' + content3)
                f.reset()
                readbytes = f.read(4096)
                self.assertEqual(readbytes, content1[:4096])
                readbytes = f.read(4096)
                self.assertEqual(readbytes, content1[4096:8192])
        with self.subTest(msg='test seek and tell'):
            with UniIO(os.path.join(pathprefix, filename + '*'), 'rb') as f:
                self.assertEqual(f.tell(), 0)
                readbytes = f.read(4096)
                self.assertEqual(f.tell(), 4096)
                f.seek(-4096, os.SEEK_END)
                self.assertEqual(f.tell(), 65536 + 1 + 65536 + 1 + 65536 - 4096)
                readbytes = f.read()
                self.assertEqual(readbytes, content3[-4096:])
                f.seek(65536 + 1 + 4096, os.SEEK_SET)
                readbytes = f.read(65536 + 1)
                self.assertEqual(readbytes, content2[4096:] + b'\n' + content3[:4096])
                f.seek(-8192, os.SEEK_CUR)
                readbytes = f.read(8192)
                self.assertEqual(readbytes, content2[-4095:] + b'\n' + content3[:4096])
                f.seek(4096, os.SEEK_CUR)
                readbytes = f.read(1)
                self.assertEqual(readbytes, content3[8192:8193])

    def _test_pickle(self, pathprefix):
        # Save a numpy array with pickle.dump, and then reload it with pickle.load. 
        filename = 'unit_test.arr.pkl'
        arr = numpy.random.randn(256, 256)
        with UniIO(os.path.join(pathprefix, filename), 'wb') as f:
            pickle.dump(arr, f)
        time.sleep(DELAY)
        with UniIO(os.path.join(pathprefix, filename), 'rb') as f:
            arr2 = pickle.load(f)
        self.assertEqual(numpy.sum(arr == arr2), 65536)

    def _test_torch_save(self, pathprefix):
        # Save a model, and then reload it from the checkpoint. 
        filename = 'unit_test.model.checkpoint'
        config = {
            'class': 'Seq2Seq',
            'encoder': {
                'class': 'TransformerEncoder',
            },
            'decoder': {
                'class': 'TransformerDecoder'
            },
            'share_embedding': 'decoder-input-output',
            'd_model': 512,
        }
        model = create_model(config)
        with UniIO(os.path.join(pathprefix, filename), 'wb') as f:
            torch.save(model, f)
        time.sleep(DELAY)
        with UniIO(os.path.join(pathprefix, filename), 'rb') as f:
            model2 = torch.load(f)

        def compare_weight(m1, m2):
            for p1, p2 in zip(m1.named_parameters(), 
                              m2.named_parameters()):
                if not p1[0] == p2[0]:
                    return False
                if not torch.equal(p1[1].data, p2[1].data):
                    return False
            return True
        
        self.assertEqual(compare_weight(model, model2), True)


if __name__ == '__main__':
    unittest.main()      
