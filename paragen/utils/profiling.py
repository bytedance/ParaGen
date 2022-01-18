import time
from contextlib import contextmanager
from tabulate import tabulate
import numpy as np
from typing import List
import logging
logger = logging.getLogger(__name__)

from paragen.utils.runtime import Environment


class ram:
    '''
    Code adapted from: https://github.com/dugu9sword/lunanlp/blob/master/lunanlp/ram.py

    The ram system is used to conveniently create globally temporary values in
    any place of a code.
    The values to store in a ram have the below features:
        - Users do not want to **declare it explicitly** in the program, which
            makes the code rather dirty.
        - Users want to **share** it across functions, or even files.
        - Users use it **temporarily**, such as for debugging
        - Users want to **reuse** a group of values several times, while **reset** each
            value in the group before reusing them will add a great overhead to the code.
    '''
    
    _memory = {}

    @staticmethod
    def list_keys(prefix=None):
        if prefix is None:
            return sorted(list(ram._memory.keys()))
        else:
            return sorted([ele for ele in ram._memory if ele.startswith(prefix)])

    @staticmethod
    def write(k, v):
        ram._memory[k] = v

    @staticmethod
    def pop(k):
        return ram._memory.pop(k)

    @staticmethod
    def stack_push(k, v):
        if k not in ram._memory:
            ram._memory[k] = []
        ram._memory[k].append(v)

    @staticmethod
    def stack_pop(k):
        return ram._memory[k].pop()

    @staticmethod
    def inc(k):
        if k not in ram._memory:
            ram._memory[k] = 0
        ram._memory[k] = ram._memory[k] + 1

    @staticmethod
    def read(k):
        return ram._memory[k]

    @staticmethod
    def has(k):
        return k in ram._memory

    @staticmethod
    def flag_name(k):
        return f'ram_FLAG_{k}'

    @staticmethod
    def set_flag(k):
        ram.write(ram.flag_name(k), True)

    @staticmethod
    def reset_flag(k):
        if ram.has(ram.flag_name(k)):
            ram.pop(ram.flag_name(k))

    @staticmethod
    def has_flag(k, verbose_once=False):
        ret = ram.has(ram.flag_name(k)) and ram.read(ram.flag_name(k)) is True
        if verbose_once and not ram.has_flag(f'VERBOSE_ONCE_{ram.flag_name(k)}'):
            print(
                f'INFO: check the flag {k}={ret}, the information only occurs once.'
            )
            ram.set_flag(f'VERBOSE_ONCE_{ram.flag_name(k)}')
        return ret

    @staticmethod
    def reset(prefix=None):
        if prefix is not None:
            to_reset = []
            for key in ram._memory:
                if key.startswith(prefix):
                    to_reset.append(key)
            for key in to_reset:
                ram._memory.pop(key)
        else:
            ram._memory.clear()


class profiler:
    
    @staticmethod
    def window():
        return Environment().profiling_window

    PREFIX = '__PROFILER__@'
    RECORDS_PREFIX = '__PROFILER__@__RECORD__@'
    TIMEIT_STACK = '__PROFILER__@__TIMEIT_STACK__'

    _time = 0
    _cycle_start_time = 0.0

    @staticmethod
    def cycle_start():
        if profiler.window() == 0:
            return

        profiler._time += 1
        profiler._cycle_start_time = time.time()

    @staticmethod
    def cycle_end():
        if profiler.window() == 0:
            return

        ram.stack_push(f'{profiler.RECORDS_PREFIX}*total', time.time() - profiler._cycle_start_time)
        if profiler._time % profiler.window() == 0:
            table = [('name', 'num calls', 'secs', 'secs/call')] # type: list[object]
            total_cost = 0.0
            saved_cost = 0.0
            for ele in ram.list_keys(profiler.RECORDS_PREFIX):
                key = ele.replace(profiler.RECORDS_PREFIX, '')
                values: List[float] = ram.read(ele)
                table.append((key, len(values), np.sum(values), np.mean(values)))
                if key == '*total':
                    total_cost = np.sum(values)
                elif '.' not in key:
                    saved_cost += np.sum(values)
            table.append(('*rest', '-', total_cost - saved_cost, '-'))
            logger.info('\n' + tabulate(table))
            ram.reset(profiler.PREFIX)

    @staticmethod
    @contextmanager
    def timeit(sth):
        if profiler.window() == 0:
            yield 
            return

        assert sth not in [ '*total' , '*rest', '' ]
        assert '.' not in sth, 'Do not use `.` in profiler!'
        start = time.time()
        ram.stack_push(f'{profiler.TIMEIT_STACK}', sth)
        yield
        prof_key = '.'.join(ram.read(profiler.TIMEIT_STACK))

        ram.stack_pop(profiler.TIMEIT_STACK)

        end = time.time()
        ram.stack_push(f'{profiler.RECORDS_PREFIX}{prof_key}', end - start)
