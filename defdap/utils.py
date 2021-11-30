# Copyright 2021 Mechanics of Microstructures Group
#    at The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from datetime import datetime


def reportProgress(message: str = ""):
    """Decorator for reporting progress of given function

    Parameters
    ----------
    message
        Message to display (prefixed by 'Starting ', progress percentage
        and then 'Finished '

    References
    ----------
    Inspiration from :
    https://gist.github.com/Garfounkel/20aa1f06234e1eedd419efe93137c004

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            messageStart = f"\rStarting {message}.."
            print(messageStart, end="")
            # The yield statements in the function produces a generator
            generator = func(*args, **kwargs)
            progPrev = 0.
            printFinal = True
            ts = datetime.now()
            try:
                while True:
                    prog = next(generator)
                    if type(prog) is str:
                        printFinal = False
                        print("\r" + prog)
                        continue
                    # only report each percent
                    if prog - progPrev > 0.01:
                        messageProg = f"{messageStart} {prog*100:.0f} %"
                        print(messageProg, end="")
                        progPrev = prog
                        printFinal = True

            except StopIteration as e:
                if printFinal:
                    te = str(datetime.now() - ts).split('.')[0]
                    messageEnd = f"\rFinished {message} ({te}) "
                    print(messageEnd)
                # When generator finished pass the return value out
                return e.value

        return wrapper
    return decorator


class Datastore(object):
    __slots__ = ['_store', '_generators']

    def __init__(self):
        self._store = {}
        self._generators = {}

    def __len__(self):
        return len(self._store)

    def __str__(self):
        text = 'Datastore'
        for key, val in self._store.items():
            text += f'\n  {key}: {val["data"].__repr__()}'

        return text

    def __contains__(self, key):
        return key in self._store

    def __getitem__(self, key):
        if isinstance(key, tuple):
            attr = key[1]
            key = key[0]
        else:
            attr = 'data'

        if key not in self:
            raise ValueError(f'Data with key `{key}` does not exist.')
        if attr not in self._store[key]:
            raise ValueError(f'Metadata `{attr}` does not exist for `{key}`.')

        val = self._store[key][attr]

        # Generate data if needed
        if attr == 'data' and val is None:
            try:
                self.generate(key)
            except ValueError:
                # No generator found
                return val

            val = self._store[key][attr]

        return val

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            attr = key[1]
            key = key[0]
        else:
            attr = 'data'

        self._store[key][attr] = val

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in ('_store', '_generators'):
            super().__setattr__(key, val)
        else:
            self[key] = val

    def __iter__(self):
        for key in self.keys():
            yield key

    def keys(self):
        return self._store.keys()

    # def values(self):
    #     return self._store.values()

    # def items(self):
    #     return dict(**self)

    def add(self, key, data, **kwargs):
        if key in self:
            raise ValueError(f'Data with key `{key}` already exists.')
        if 'data' in kwargs:
            raise ValueError(f'Metadata name `data` is not allowed.')

        self._store[key] = {
            'data': data,
            **kwargs
        }

    def add_generator(self, keys, func, metadatas=None, **kwargs):
        """

        Parameters
        ----------
        keys: str or tuple of str
        func: callable
        kwargs

        Returns
        -------

        """
        if isinstance(keys, str):
            keys = (keys, )
        if isinstance(metadatas, dict):
            metadatas = (metadatas, )
        for i, key in enumerate(keys):
            if metadatas is None:
                metadata = {}
            else:
                metadata = metadatas[i]
            metadata.update(kwargs)
            self.add(key, None, **metadata)
        self._generators[keys] = func

    def generate(self, key):
        for (keys, generator) in self._generators.items():
            if key in keys:
                datas = generator()
                if len(keys) == 1:
                    self[key] = datas
                    break
                for key, data in zip(keys, datas):
                    self[key] = data
                break
        else:
            ValueError(f'Generator not found for data `{key}`')

    def update(self, other, priority='other'):
        if priority == 'self':
            other._store.update(self._store)
            self._store = other._store
        else:
            self._store.update(other._store)

    def get_metadata(self, key, attr, value=None):
        return self._store[key].get(attr, value)
