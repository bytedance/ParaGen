from io import IOBase, TextIOBase
from multiprocessing import Process
import os
import sys
import re
import time
import json
import subprocess
import random
import logging
logger = logging.getLogger(__name__)

from paragen.utils.runtime import Environment


SPACE_NORMALIZER = re.compile("\s+")
TEMP_IO_SAVE_PATH = ""


def init_io():
    global TEMP_IO_SAVE_PATH
    try:
        TEMP_IO_SAVE_PATH = os.path.join(os.getenv('HOME'), '.cache/uio/')
    except Exception:
        TEMP_IO_SAVE_PATH = os.path.join(os.getcwd(), '.cache_uio/')
    if not os.path.exists(TEMP_IO_SAVE_PATH):
        os.makedirs(TEMP_IO_SAVE_PATH, exist_ok=True)


def clear_cache():
    global TEMP_IO_SAVE_PATH
    output = subprocess.run('lsof +d {}'.format(TEMP_IO_SAVE_PATH).split(), capture_output=True)
    occupied = str(output.stdout, encoding='utf8').split('\n')
    occupied = set([filepath for filepath in occupied if filepath])
    for name in os.listdir(TEMP_IO_SAVE_PATH):
        filename = os.path.join(TEMP_IO_SAVE_PATH, name)
        if filename not in occupied:
            try:
                os.remove(filename)
            except:
                pass


init_io()


def _run_cmd(args_list):
    """
    run linux commands
    """
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return =  proc.returncode
    return s_return, s_output, s_err

def parse_single_path(path):
    """
    Parse path with regular expression

    Args:
        path: input path

    Returns:
        - parse path list
    """
    def _get_files(path):
        return [f for f in listdir(path, return_files=True, return_dirs=False)]

    if path.endswith('*'):
        path = path.split('/')
        pathdir, pathprefix = '/'.join(path[:-1]), path[-1][:-1]
        files = ['{}/{}'.format(pathdir, f) for f in _get_files(pathdir) if f.startswith(pathprefix)]
    elif isdir(path):
        files = ['{}/{}'.format(path, f) for f in _get_files(path)]
    else:
        files = [path]
    random.shuffle(files)
    return files


def parse_path(path):
    files = []
    for singlepath in path.strip().split(','):
        if singlepath:
            files += parse_single_path(singlepath)
    return files


def read_vocab(path):
    """
    Read a vocab

    Args:
        path: path to restore vocab

    Returns:
        - a dict of frequency table
    """
    freq = []
    with UniIO(path, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            line = SPACE_NORMALIZER.split(line)
            freq.append((' '.join(line[:-1]), int(line[-1])))
    return freq


def read_table(path):
    """
    Read a table

    Args:
        path: path to restore table

    Returns:
        - a dict of table
    """
    table = {}
    with UniIO(path, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            line = SPACE_NORMALIZER.split(line)
            table[' '.join(line[:-1])] = line[-1]
    return table


def read_list(path):
    """
    Read a list

    Args:
        path: path to restore list

    Returns:
        - a list
    """
    with UniIO(path, 'r') as fin:
        freq = [line.strip('\n') for line in fin]
    return freq


def jsonable(x):
    """
    Check if x is suit json.dumps
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def listdir(path, return_files=True, return_dirs=False, retry=5):
    """
    Given a path, return a list of files under this path

    :param path: directory
    :return: a list of files / dirs
    """
    def _listdir(path):
        retval = list()
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run('hadoop fs -ls {}'.format(path).split(), capture_output=True)
                returncode = output.returncode
                output = output.stdout
                output = str(output, encoding='utf8').split('\n')
                getname = lambda x: x.split('/')[-1]
                if return_files:
                    retval += [getname(f) for f in output if f.startswith('-')]
                if return_dirs:
                    retval += [getname(f) for f in output if f.startswith('d')]
            else:
                output = subprocess.run('ls -A -H -l {}'.format(path).split(), capture_output=True)
                returncode = output.returncode
                output = output.stdout
                output = str(output, encoding='utf8').split('\n')
                getname = lambda x: x.split(' ')[-1]
                if return_files:
                    retval += [getname(f) for f in output if f.startswith('-')]
                if return_dirs:
                    retval += [getname(f) for f in output if f.startswith('d')]
            if returncode == 0:
                break
        if returncode != 0:
            logger.warning(f'fail to listdir {path}')
        return retval

    if path:
        return _listdir(path)
    else:
        raise ValueError


def isdir(path):
    """
    Check if a path if a directory

    :param path: path to check
    :return:
    """
    if path.startswith('hdfs:'):
        output = subprocess.run('hadoop fs -test -d {}'.format(path).split(), capture_output=True)
        return output.returncode == 0
    else:
        return os.path.isdir(path)


def wait_until_exist(path, timeout=10000):
    start = time.time()
    while True:
        if exists(path):
            return True
        if time.time() - start > timeout:
            logger.warning(f"timeout: {path} not found!")
            return False
        time.sleep(5)


def cp(src, tgt, retry=5, wait=False):
    """
    Copy a file from src to tgt

    :param src: source file / directory
    :param tgt: target file / directory
    :return:
    """
    def _cp(src, tgt):
        if not wait_until_exist(src):
            logger.info(f'timeout: {src} not found')
            return
        returncode = 1
        for i in range(retry):
            if exists(tgt):
                remove(tgt, wait=True)
            if src.startswith('hdfs:') and tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-cp", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif src.startswith('hdfs:') and not tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-get", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif not src.startswith('hdfs:') and tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-put", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(["cp", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully copy from {src} to {tgt}')
                break
        if returncode != 0:
            logger.warning(f'copy from {src} to {tgt} fails')

    env = Environment()
    if env.is_master():
        if wait:
            _cp(src, tgt)
        else:
            Process(target=_cp, args=(src, tgt)).start()


def mkdir(path, retry=5, wait=True):
    """
    Create a directory at path

    :param path: path to directory
    :return:
    """
    def _mkdir(path):
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-mkdir", "-p", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(["mkdir", "-p", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully make directory: {path}')
                break
        if returncode != 0:
            logger.warning(f'mkdir {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _mkdir(path)
        else:
            Process(target=_mkdir, args=(path,)).start()


def remove(path, retry=5, wait=False):
    """
    Remove a directory or file

    :param path: path to remove
    :return:
    """
    def _remove(path):
        if exists(path):
            returncode = 1
            for i in range(retry):
                if path.startswith('hdfs:'):
                    output = subprocess.run(['hadoop', 'fs', '-rm', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    output = subprocess.run(['rm', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                returncode = output.returncode
                if returncode == 0:
                    logger.info(f'successfully remove file: {path}')
                    break
            if returncode != 0:
                logger.warning(f'remove file {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _remove(path)
        else:
            Process(target=_remove, args=(path,)).start()


def exists(path):
    """
    check if path exists

    :param path: path to check
    :return:
    """
    if path.startswith('hdfs:'):
        r = subprocess.run(['hadoop', 'fs', '-stat', path], capture_output=True)
        return True if r.returncode == 0 else False
    else:
        return os.path.exists(path)


def not_exist(paths):
    for p in paths:
        if not exists(p):
            return p
    return None


def remove_tree(path, retry=5, wait=True):
    """
    remove directory recursively

    :param path: path to remove
    :return
    """
    def _rmtree(path):
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run(['hadoop', 'fs', '-rm', '-r', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(['rm', '-r', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully remove directory: {path}')
                break
        if returncode != 0:
            logger.warning(f'remove directory {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _rmtree(path)
        else:
            Process(target=_rmtree, args=(path,)).start()


def create_data_map(path):
    """
    read a data map from path
    """
    data_map = []
    with UniIO(path) as fin:
        data_position = 0
        for i, line in enumerate(fin):
            d = json.loads(line)
            token_num = d['token_num'] if 'token_num' in d else 1
            data_map.append((i, data_position, token_num))
            data_position += len(line)
    return data_map


def utf8len(s):
    """
    Get the byte number of the utf-8 sentence.
    """
    return len(s.encode('utf-8'))


def _InputFileOpen(path, mode='r', encoding='utf8', timeout=-1, poll_interval=0.1, *args, **kwargs):
    try:
        if path.startswith('hdfs:'):
            if 'localpath' in kwargs:
                localpath = kwargs['localpath']
            else:
                localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', path)
            lockfilename = localpath + '.lock'  # Multiprocess may read the file; they share the same cached file; 
                                                # They need to wait until it is downloaded completely
            if not os.path.exists(lockfilename):  # acquire lock
                fd = os.open(lockfilename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)  # lock
                if os.path.exists(localpath):
                    os.remove(localpath)
                p = subprocess.run("hadoop fs -get {} {}".format(path, localpath).split(),
                                   capture_output=True)
                if p.returncode:
                    logger.warning(f'failed to open {path}, hadoop fs return code: {p.returncode}')
                os.close(fd)
                os.remove(lockfilename)  # release lock
            else:
                start = time.time()
                while True:  # Wait until the file is released (finished downloading)
                    if not os.path.exists(lockfilename):
                        break
                    if timeout >= 0 and time.time() - start > timeout:
                        logger.warning(f'failed to open {path}, file is locked, timeout')
                        break
                    time.sleep(poll_interval)
        else:
            localpath = path
        if 'b' in mode.lower():
            istream = open(localpath, mode=mode)
        else:
            istream = open(localpath, mode=mode, encoding=encoding)
        # logger.info(f'successfully open file: {path}')
        return istream
    except Exception as e:
        logger.warning(f'open file {path} fails: {e}')
        return None


class _InputStream(TextIOBase):
    """
    A InputSteam wrapper to tackle with multiple files input
    """
    def __init__(self, path, encoding='utf8'):
        super().__init__()
        self._paths = parse_path(path)
        _hash = hash(''.join(self._paths + [str(os.getpid())]))
        _hash &= sys.maxsize
        self._localpath = os.path.join(TEMP_IO_SAVE_PATH, str(_hash))
        self._encoding = encoding
        self._idx = -1
        self._fin = None
        self._next_file()
    
    def _next_file(self):
        if self._fin is not None:
            self._fin.close()
        self._idx += 1
        if 0 <= self._idx < len(self._paths):
            self._fin = _InputFileOpen(self._paths[self._idx], mode='r', encoding=self._encoding, localpath=self._localpath)
            if self._fin is None:
                self._next_file()
        else:
            raise StopIteration

    def reset(self):
        self._idx = -1
        self._next_file()

    def close(self):
        if self._fin is not None:
            self._fin.close()
        super().close()
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self._idx >= len(self._paths):
               raise IndexError
            return next(self._fin)
        except StopIteration:
            try:
                self._next_file()
                return self.__next__()
            except Exception as e:
                raise e
        except IndexError:
            raise StopIteration

    def readline(self, size=-1):
        if self._fin is None or self._fin.closed:
            return ''
        sample = self._fin.readline(size)
        if sample:
            return sample
        try:
            self._next_file()
            return self.readline(size)
        except StopIteration:
            return ''

    def readlines(self, hint=-1):
        retval = []
        total_size = 0
        while hint is None or hint <= 0 or total_size <= hint:
            line = self.readline()
            if line:
                retval.append(line)
                total_size += len(line)
            else:
                break
        return retval

    def read(self, size=-1):
        if self._fin is None or self._fin.closed:
            return ''
        if size == -1:
            buffer = ''
            while True:
                buffer += self._fin.read()
                try:
                    self._next_file()
                except StopIteration:
                    break
            return buffer
        else:
            buffer = ['' for i in range(size)]
            offset = 0
            while size > 0:
                filesize = self._size(self._fin)
                if filesize <= size:
                    buffer[offset : offset + filesize] = self._fin.read()
                    offset += filesize
                    size -= filesize
                    try:
                        self._next_file()
                    except StopIteration:
                        break
                else:
                    buffer[offset : ] = self._fin.read(size)
                    size = 0
            buffer = ''.join(buffer)
            return buffer
    
    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            if offset < 0:
                raise OSError(22, 'Invalid argument')
            self.reset()
            _offset = offset
            while offset > 0:
                size = self._size(self._fin)
                if offset <= size:
                    self._fin.seek(offset, os.SEEK_SET)
                    offset = 0
                else:
                    offset -= size
                    try:
                        self._next_file()
                    except StopIteration:
                        break 
            return _offset
        elif whence == os.SEEK_CUR:
            if offset:
                raise ValueError(f'invalid offset {offset}, offset must be zero')
            else:
                pass  # do nothing, according to TextIOBase.seek()
            return self.tell()
        elif whence == os.SEEK_END:
            if offset:
                raise ValueError(f'invalid offset {offset}, offset must be zero')
            else:
                while True:
                    try:
                        self._next_file()
                    except StopIteration:
                        break
            return self.tell()
        else:
            raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')
            
    def tell(self):
        return self._fin.tell()  # Not a proper implementation

    def _size(self, fin):
        cur = fin.tell()
        tail = fin.seek(0, os.SEEK_END)
        size = max(0, tail - cur)
        fin.seek(cur, os.SEEK_SET)
        return size


def _OutputFileOpen(path, localpath, mode='w', encoding='utf8'):
    try: 
        if path.startswith('hdfs:'):
            if not os.path.exists(TEMP_IO_SAVE_PATH):
                os.mkdir(TEMP_IO_SAVE_PATH)
        else:
            localpath = path
        if 'b' in mode.lower():
            ostream = open(localpath, mode=mode)
        else:
            ostream = open(localpath, mode=mode, encoding=encoding)
        return ostream
    except Exception as e:
        logger.warning(f'open file {path} fails: {e}')


class _OutputStream(TextIOBase):
    """
    OutputStream is an io wrapper to tackle with multiple kinds of path

    Args:
        path: output file path
    """
    def __init__(self, path, encoding='utf8'):
        super().__init__()
        self._path = path
        if self._path.startswith('hdfs:'):
            self._localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', '{}_{}_w'.format(path, os.getpid()))
        else:
            self._localpath = path
        self._encoding = encoding
        self._fout = _OutputFileOpen(path, self._localpath, encoding=encoding)

    def reset(self):
        """
        Reset output stream
        """
        self._fout.seek(0)
    
    def close(self):
        """
        Close output stream
        """
        self._fout.close()
        if self._path.startswith('hdfs:'):
            cp(self._localpath, self._path, wait=True)
            wait_until_exist(self._path)
        super().close()

    def write(self, content):
        """
        Write to output stream

        Args:
            content: str to write
        """
        self._fout.write(content)
    
    def writelines(self, content):
        """
        Write to output InputStream

        Args:
            content: list of str
        """
        self._fout.writelines(content)

    def seek(self, offset, whence=os.SEEK_SET):
        """
        The same as TextIOBase.seek()
        """
        return self._fout.seek(offset, whence)

    def tell(self):
        """
        The same as TextIOBase.tell()
        """
        return self._fout.tell()


class _InputBytes(IOBase):
    """
    InputBytes is an io wrapper to tackle with multiple kinds of path

    Args:
        path: input file path
    """
    def __init__(self, path, mode='rb'):
        super().__init__()
        self._paths = parse_path(path)
        self._fins = [_InputFileOpen(path, mode=mode) for path in self._paths]
        self._fins = [item for item in self._fins if item is not None]
        self._sizes = [self._size(fin) for fin in self._fins]
        self._idx = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Fetch next line from file.  The line terminator is b'\n'.

        Returns:
            - next line
        """
        try:
            if self._idx >= len(self._fins):
                raise IndexError
            sample = next(self._fins[self._idx])
            return sample
        except StopIteration:
            self._idx += 1
            sample = self.__next__()
            return sample
        except IndexError:
            raise StopIteration

    def reset(self):
        """
        Reset input stream
        """
        self._idx = 0
        for fin in self._fins:
            fin.seek(0)

    def readline(self, size=-1):
        """
        Read the next line.  Return b'' at EOF.  The line terminator is b'\n'.

        Args:
            size: read at most `size` bytes

        Returns:
            - next line  
        """
        try:
            if size == 0:
                return b''
            if self._idx >= len(self._fins):
                raise StopIteration
            sample = self._fins[self._idx].readline(size)
            if sample:
                return sample
            self._idx += 1
            return self.readline(size)
        except StopIteration:
            return b''
    
    def readlines(self, hint=-1):
        """
        Read all lines and return in a list
        
        Args:
            hint: read at most `hint` bytes

        Returns:
            - list of lines
        """
        retval = []
        total_size = 0
        while hint is None or hint <= 0 or total_size <= hint:
            line = self.readline()
            if line:
                retval.append(line)
                total_size += len(line)
            else:
                break
        return retval

    def read(self, size=-1):
        """
        Read the rest of file

        Args:
            size: read at most `size` bytes

        Returns:
            - the rest of file
        """
        if size == -1:
            buffer = b''
            while self._idx < len(self._fins):
                buffer += self._fins[self._idx].read()
                self._idx += 1
            return buffer
        else:
            buffer = bytearray(size)
            offset = 0
            while self._idx < len(self._fins) and size > 0:
                filesize = self._size(self._fins[self._idx])
                if filesize <= size:
                    buffer[offset : offset + filesize] = self._fins[self._idx].read()
                    offset += filesize
                    self._idx += 1
                    size -= filesize
                else:
                    buffer[offset : ] = self._fins[self._idx].read(size)
                    size = 0
            buffer = bytes(buffer)
            return buffer
                            
    def _size(self, fin):
        # Given a file descriptor, calculate its size
        cur = fin.tell()
        tail = fin.seek(0, os.SEEK_END)
        size = max(0, tail - cur)
        fin.seek(cur, os.SEEK_SET)
        return size

    def tell(self):
        """
        Return the absolute current stream position

        Returns:
            - current stream position
        """
        position = 0
        if self._idx < len(self._fins):
            position += self._fins[self._idx].tell()
        for i in range(min(self._idx, len(self._fins))):
            position += self._sizes[i]
        return position

    def seek(self, offset, whence=os.SEEK_SET):
        """
        Change the stream position to the given byte offset.

        Args:
            offset: byte offset
            whence: Values for whence are SEEK_SET (0), SEEK_CUR (1) or SEEK_END (2)
        
        Returns:
            Stream position after seek
        """
        if whence == os.SEEK_SET:
            if offset < 0:
                raise OSError(22, 'Invalid argument')
            return self.seek(offset - self.tell(), whence=os.SEEK_CUR)
        if whence == os.SEEK_CUR:
            self._idx = max(0, min(len(self._fins) - 1, self._idx))
            while self._idx < len(self._fins) and offset > 0:
                filesize = self._size(self._fins[self._idx])
                if filesize < offset:
                    self._fins[self._idx].seek(0, os.SEEK_END)
                    self._idx += 1
                    offset -= filesize
                else:
                    self._fins[self._idx].seek(offset, os.SEEK_CUR)
                    offset = 0
            while self._idx >= 0 and offset < 0:
                filesize = self._fins[self._idx].tell()
                if offset + filesize < 0:
                    self._fins[self._idx].seek(0, os.SEEK_SET)
                    self._idx -= 1
                    offset += filesize
                else:
                    self._fins[self._idx].seek(offset, os.SEEK_CUR)
                    offset = 0
            self._idx = max(0, min(len(self._fins) - 1, self._idx))
            return self.tell()
        if whence == os.SEEK_END:
            for i in range(len(self._fins)):
                offset += self._sizes[i]
            return self.seek(offset, whence=os.SEEK_SET)
        raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')

    def close(self):
        """
        Close the input stream
        """
        for fin in self._fins:
            fin.close()
        super().close()


class _OutputBytes(IOBase):
    """
    OutputBytes is an io wrapper to tackle with multiple kinds of path

    Args:
        path: output file path
    """
    def __init__(self, path, mode='wb'):
        super().__init__()
        self._path = path
        self._localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', '{}_{}_w'.format(path, os.getpid()))
        self._fout = _OutputFileOpen(path, self._localpath, mode=mode)

    def reset(self):
        """
        Reset output stream
        """
        self._fout.seek(0)
    
    def close(self):
        """
        Close output stream
        """
        self._fout.close()
        if self._path.startswith('hdfs:'):
            cp(self._localpath, self._path, wait=True)
            wait_until_exist(self._path)
        super().close()

    def write(self, content):
        """
        Write to output Stream

        Args:
            content: bytes to write
        """
        self._fout.write(content)

    def seek(self, offset, whence=os.SEEK_SET):
        """
        The same as IOBase.seek()
        """
        return self._fout.seek(offset, whence)

    def tell(self):
        """
        The same as IOBase.tell()        
        """
        return self._fout.tell()


class UniIO(_InputStream, _OutputStream, _InputBytes, _OutputBytes):
    """
    A universal IO with the same functions as python:open
    """
    def __init__(self, path, mode='r', encoding='utf8'):
        pass

    def __new__(cls, path, mode='r', encoding='utf8'):
        if 'r' in mode.lower():
            if 'b' in mode.lower():
                return _InputBytes(path, mode=mode)
            return _InputStream(path, encoding=encoding)
        elif 'w' in mode.lower():
            if 'b' in mode.lower():
                return _OutputBytes(path, mode=mode)
            return _OutputStream(path, encoding=encoding)
        logger.warning(f'Not support file mode: {mode}')
        raise ValueError

