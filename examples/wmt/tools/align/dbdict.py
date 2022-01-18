# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

'''
A dictionary-like object with SQLite backend
============================================
Python dictionaries are very efficient objects for fast data access. But when
data is too large to fit in memory, you want to keep data on disk but available
for fast random access.
Here's a dictionary-like object which uses a SQLite database backend for random
access to the dictionary's key-value pairs.
Use it like a standard dictionary, except that you give it a name
(eg.'tempdict'):
    import dbdict
    d = dbdict.open('tempdict')
    d['foo'] = 'bar'
    # At this point, the key value pair foo and bar is written to disk.
    d['John'] = 'doh!'
    d['pi'] = 3.999
    d['pi'] = 3.14159  # replaces the previous version of pi
    d['pi'] += 1
    d.close()    # close the database file
You can access your dictionary later on:
    d = dbdict.open('tempdict')
    del d['foo']
    if 'John' in d:
        print 'John is in there !'
    print d.items()
For efficient inserting/updating a list of key-value pairs, use the update()
method:
    d.update([('f1', 'test'), ('f2', 'example')])
    d.update({'f1':'test', 'f2':'example'})
    d.update(f1='test', f2='example')
Use get() method to most efficiently get a number of items as specified by a
lis of keys:
    d.get(['f1', 'f2'])
Use remove() method to most efficiently remove a number of items as specified
by a list of keys:
    d.remove(['f1', 'f2'])
There is also an alternative (fully equivalent) way to instanstiate a dbdict
object by the call:
    from dbdict import dbdict
    d = dbdict('tempdict')
Make a memory-based (ie not filed based) SQLite database by the call:
    dbdict(':memory:')
Other special functionality as compared to dict:
    d.clear()    Clear all items (and free up unused disk space)
    d.reindex()  Delete and recreate the key index
    d.vacuum()   Free up unused disk space
    d.con        Access to the underlying SQLite connection (for advanced use)
Some things to note:
  - You can't directly store Python objects. Only numbers, strings and binary
    data. Objects need to be serialized first in order to be stored. Use e.g.
    pickle, json (or simplejson) or yaml for that purpose.
  - Explicit database connection closing using the close() method is not
    required. Changes are written on key-value assignment to the dictionary.
    The file stays open until the object is destroyed or the close() method is
    called.
    Original code by Jacob Sondergaard
        hg clone ssh://hg@bitbucket.org/nephics/dbdict
    Modified to pickle automatically
'''

__version__ = '1.3.1'

import sqlite3

try:
    from collections.abc import MutableMapping
except ImportError:
    # MutableMapping is new in Python 2.6+
    from collections import MutableMapping

from os import path

try:
    import cPickle as pickle
except ImportError:
    import pickle


class DbDict(MutableMapping):
    ''' DbDict, a dictionary-like object with SQLite back-end '''

    def __init__(self, filename, picklevalues=False):
        self.picklevalues = picklevalues
        if filename == ':memory:' or not path.isfile(filename):
            self.con = sqlite3.connect(filename)
            self._create_table()
        else:
            self.con = sqlite3.connect(filename)

    # _____________________________________________________________________________________

    #   Add automatic pickling and unpickling

    def pickle_loads(self, value):
        """
        pickle.load if specified
        """
        if self.picklevalues:
            value = pickle.loads(bytes(value))
        return value

    def pickle_dumps(self, value):
        """
        pickle.load if specified
        """
        if self.picklevalues:
            #
            #   Protocol =  0 generates ASCII 7 bit strings and is less efficient
            #   Protocol = -1 generates ASCII 8 bit strings which need to be handled as
            #     blobs by sqlite3.
            #
            #   Unfortunately, sqlite3 only understands memoryview objects in python3 and
            #   buffer objects in python2
            #
            #   http://bugs.python.org/issue7723 suggests there is no portable
            #       python2/3 way to write blobs to Sqlite
            #
            #   However, sqlite3.Binary seems to do the trick
            #
            #   Otherwise, to use protocol -1, we need to use the following code:
            #
            # if sys.hexversion >= 0x03000000:
            #    value = memoryview(pickle.dumps(value, protocol = -1))
            # else:
            #    value = buffer(pickle.dumps(value, protocol = -1))
            #
            value = sqlite3.Binary(pickle.dumps(value, protocol=-1))
        return value

    # _____________________________________________________________________________________

    def _create_table(self):
        '''Creates an SQLite table 'data' with the columns 'key' and 'value'
        where column 'key' is the table's primary key.
        Note: SQLite automatically creates an unique index for the 'key' column.
        The index may get fragmented with lots of insertions/updates/deletions
        therefore it is recommended to use reindex() when searches becomes
        gradually slower.
        '''
        self.con.execute('create table data (key PRIMARY KEY,value)')
        self.con.commit()

    def __getitem__(self, key):
        '''Return value for specified key'''
        row = self.con.execute('select value from data where key=?',
                               (key,)).fetchone()
        if not row:
            raise KeyError(key)
        return self.pickle_loads(row[0])

    def __setitem__(self, key, value):
        '''Set value at specified key'''
        value = self.pickle_dumps(value)
        self.con.execute('insert or replace into data (key, value) '
                         'values (?,?)', (key, value))
        self.con.commit()

    def __delitem__(self, key):
        '''Delete item (key-value pair) at specified key'''
        if key in self:
            self.con.execute('delete from data where key=?', (key,))
            self.con.commit()
        else:
            raise KeyError

    def __iter__(self):
        '''Return iterator over keys'''
        return self._iterquery(self.con.execute('select key from data'),
                               single_value=True)

    def __len__(self):
        '''Return the number of stored items'''
        cursor = self.con.execute('select count() from data')
        return cursor.fetchone()[0]

    @staticmethod
    def _iterquery(cursor, single_value=False):
        '''Return iterator over query result with pre-fetching of items in
        set sizes determined by SQLite backend'''
        rows = True
        while rows:
            rows = cursor.fetchmany()
            for row in rows:
                if single_value:
                    yield row[0]
                else:
                    yield row

    def iterkeys(self):
        '''Return iterator of all keys in the database'''
        return self.__iter__()

    def itervalues(self):
        '''Return iterator of all values in the database'''
        it = self._iterquery(self.con.execute('select value from data'),
                             single_value=True)
        return iter(self.pickle_loads(x) for x in it)

    def iteritems(self):
        '''Return iterator of all key-value pairs in the database'''
        it = self._iterquery(self.con.execute('select key, value from data'))
        return iter((x[0], self.pickle_loads(x[1])) for x in it)

    def keys(self):
        '''Return all keys in the database'''
        return [row[0]
                for row in self.con.execute('select key from data').fetchall()]

    def items(self):
        '''Return all key-value pairs in the database'''
        values = self.con.execute('select key, value from data').fetchall()
        return [(x[0], self.pickle_loads(x[1])) for x in values]

    def clear(self):
        '''Clear the database for all key-value pairs, and free up unsused
        disk space.
        '''
        self.con.execute('drop table data')
        self.vacuum()
        self._create_table()

    def _update(self, items):
        '''Perform the SQL query of updating items (list of key-value pairs)'''
        items = [(k, self.pickle_dumps(v)) for k, v in items]
        self.con.executemany('insert or replace into data (key, value)'
                             ' values (?, ?)', items)
        self.con.commit()

    def update(self, items=None, **kwds):
        '''Updates key-value pairs in the database.
        Items (key-value pairs) may be given by keyword assignments or using
        the parameter 'items' a dict or list/tuple of items.
        '''
        if isinstance(items, dict):
            self._update(list(items.items()))
        elif isinstance(items, list) or isinstance(items, tuple):
            self._update(items)
        elif items:
            # probably a generator
            try:
                self._update(list(items))
            except TypeError:
                raise ValueError(
                    'Could not interpret value of parameter `items` as a dict, list/tuple or iterator.')

        if kwds:
            self._update(list(kwds.items()))

    def popitem(self):
        '''Pop a key-value pair from the database. Returns the next key-value
        pair which is then removed from the database.'''
        res = self.con.execute('select key, value from data').fetchone()
        if res:
            key, value = res
        else:
            raise StopIteration
        del self[key]
        value = self.pickle_loads(value)
        return key, value

    def close(self):
        '''Close database connection'''
        self.con.close()

    def vacuum(self):
        '''Free unused disk space from the database file.
        The operation has no effect if database is in memory.
        Note: The operation can take some time to run (around a half second per
        megabyte on the Linux box where SQLite is developed) and it can use up
        to twice as much temporary disk space as the original file while it is
        running.
        '''
        self.con.execute('vacuum')
        self.con.commit()

    def mget(self, keys):
        '''Get item(s) for the specified key or list of keys.
        Items will be returned only for those keys that are defined. The
        function will pass silently (i.e. not raise an error) if one or more of
        the keys is not defined.'''
        try:
            keys = tuple(keys)
        except TypeError:
            # probably a single key (ie not an iterable)
            keys = (keys,)
        values = self.con.execute('select key, value from data where key in ({})'.format(
            ",".join(['?'] * len(keys))), keys).fetchall()
        return [(k, self.pickle_loads(v)) for k, v in values]

    def get(self, key):
        '''Return value for specified key'''
        row = self.con.execute('select value from data where key=?',
                               (key,)).fetchone()
        if not row:
            return None
        return self.pickle_loads(row[0])

    def remove(self, keys):
        '''Removes item(s) for the specified key or list of keys.
        The function will pass silently (i.e. not raise an error) if one or more
        of the keys is not defined.'''
        try:
            keys = tuple(keys)
        except TypeError:
            # probably a single key (ie not an iterable)
            keys = (keys,)
        self.con.execute('delete from data where key in %s' % (keys,))
        self.con.commit()

    def reindex(self):
        '''Delete and recreate key index.
        Use this function if key lookup time becomes slower. This may happen as
        the index will become fragmented with lots of
        insertions/updates/deletions.'''
        self.con.execute('reindex sqlite_autoindex_data_1')
        self.con.commit()
