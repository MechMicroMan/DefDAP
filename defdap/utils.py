# Copyright 2025 Mechanics of Microstructures Group
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
from uuid import uuid4


def report_progress(message: str = ""):
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
    """Storage of data and metadata, with methods to allow derived data
    to be calculated only when accessed.

    Attributes
    ----------
    _store : dict of dict
        Storage for data and metadata, keyed by data name. Each item is
        a dict with at least a `data` key, all other items are metadata,
        possibly including:
            type : str
                Type of data stored:
                    `map` - at least a 2-axis array, trailing axes are spatial
            order : int
                Tensor order of the data
            unit : str
                Measurement unit the data is stored in
            plot_params : dict
                Dictionary of the default parameters used to plot
    _generators: dict
        Methods to generate derived data, keyed by tuple of data names
        that the method produces.

    """
    __slots__ = [
        '_store',
        '_generators',
        '_derivatives',
        '_group_id',
        '_crop_func',
        '_mask_func'
    ]
    _been_to = None

    @staticmethod
    def generate_id():
        return uuid4()

    def __init__(self, group_id=None, crop_func=None, mask_func=None):
        self._store = {}
        self._generators = {}
        self._derivatives = []
        self._group_id = self.generate_id() if group_id is None else group_id
        self._crop_func = (lambda x, **kwargs: x) if crop_func is None else crop_func
        self._mask_func = (lambda x, **kwargs: x) if mask_func is None else mask_func

    def __len__(self):
        """Number of data in the store, including data not yet generated."""
        return len(self.keys())

    def __str__(self):
        text = 'Datastore'
        for key, val in self._store.items():
            # text += f'\n  {key}: {val["data"].__repr__()}'
            text += f'\n  {key}'
        text2 = ''
        for derivative in self._derivatives:
            for key in self.lookup_derivative_keys(derivative):
                text2 += f'\n    {key}'
        if text2 != '':
            text += '\n  Derived data:' + text2

        return text

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        """Get data or metadata

        Parameters
        ----------
        key : str or tuple of str
            Either the data name or tuple of data name and metadata name.

        Returns
        -------
        data or metadata

        """
        if isinstance(key, tuple):
            attr = key[1]
            key = key[0]
        else:
            attr = 'data'

        # Avoid looking up all keys over derivatives
        if key not in self._store:
            if key not in self:
                raise KeyError(f'Data with name `{key}` does not exist.')
            return self._get_derived_item(key, attr)
        if attr not in self._store[key]:
            raise KeyError(f'Metadata `{attr}` does not exist for `{key}`.')

        val = self._store[key][attr]

        # Generate data if needed
        if attr == 'data' and val is None:
            try:
                val = self.generate(key, return_val=True)
            except DataGenerationError:
                # No generator found
                pass

        if attr == 'data' and self.get_metadata(key, 'type') == 'map':
            if not self.get_metadata(key, 'cropped', False):
                binning = self.get_metadata(key, 'binning', 1)
                val = self._crop_func(val, binning=binning)
            if self.get_metadata(key, 'apply_mask', True):
                val = self._mask_func(val)

        return val

    def __setitem__(self, key, val):
        """Set data or metadata of item that already exists.

        Parameters
        ----------
        key : str or tuple of str
            Either the data name or tuple of data name and metadata name.
        val : any
            Value to set

        """
        if isinstance(key, tuple):
            attr = key[1]
            key = key[0]
        else:
            attr = 'data'

        if key not in self:
            raise ValueError(f'Data with name `{key}` does not exist.')

        ## TODO: fix derived data
        self._store[key][attr] = val

    def __getattr__(self, key):
        """Get data

        """
        return self[key]

    def __setattr__(self, key, val):
        """Set data of item that already exists.

        """
        if key in self.__slots__:
            super().__setattr__(key, val)
        else:
            self[key] = val

    def __iter__(self):
        """Iterate through the data names. Allows use of `*datastore` to
        get all keys in the store, imitating functionality of a dictionary.

        """
        for key in self.keys():
            yield key

    def keys(self):
        """Get the names of all data items. Allows use of `**datastore`
        to get key-value pairs, imitating functionality of a dictionary.

        """
        keys = list(self._store.keys())
        for derivative in self._derivatives:
            keys += self.lookup_derivative_keys(derivative)
        return keys

    def lookup_derivative_keys(self, derivative):
        root_call = False
        if Datastore._been_to is None:
            root_call = True
            Datastore._been_to = set()
        Datastore._been_to.add(self._group_id)

        source = derivative['source']
        matched_keys = []
        if source._group_id in Datastore._been_to:
            return matched_keys
        for key in source:
            for meta_key in derivative['in_props']:
                if source.get_metadata(key, meta_key) != derivative['in_props'][meta_key]:
                    break
            else:
                matched_keys.append(key)

        if root_call:
            Datastore._been_to = None

        return matched_keys

    def _get_derived_item(self, key, attr):
        for derivative in self._derivatives:
            if key in self.lookup_derivative_keys(derivative):
                break
            else:
                raise KeyError(f'Data with name `{key}` does not exist.')
        source = derivative['source']

        ## TODO: fix derived metadata
        # if attr not in source._store[key]:
        #     raise KeyError(f'Metadata `{attr}` does not exist for `{key}`.')

        if attr in derivative['out_props']:
            return derivative['out_props'][attr]

        if derivative['pass_ref'] and attr == 'data':
            return derivative['func'](key)

        val = derivative['source'][(key, attr)]
        if attr == 'data':
            val = derivative['func'](val)

        return val

    # def values(self):
    #     return self._store.values()

    # def items(self):
    #     return dict(**self)

    def add(self, key, data, **kwargs):
        """Add an item to the datastore.

        Parameters
        ----------
        key : str
            Name of the data.
        data : any
            Data to store.
        kwargs : dict
            Key-value pairs stored as the items metadata.

        """
        if key in self:
            raise ValueError(f'Data with name `{key}` already exists.')
        if 'data' in kwargs:
            raise ValueError(f'Metadata name `data` is not allowed.')

        self._store[key] = {
            'data': data,
            **kwargs
        }

    def add_generator(self, keys, func, metadatas=None, **kwargs):
        """Add a data generator method that produces one or more data.

        Parameters
        ----------
        keys: str or tuple of str
            Name(s) of data that the generator produces.
        func: callable
            Method that produces the data. Should return the same number
            of values as there are `keys`.
        metadatas : list of dict
            Metadata dicts for each of data items produced.
        kwargs : dict
            Key-value pairs stored as the items metadata for every data
            item produced.

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

    def add_derivative(self, datastore, derive_func, in_props=None,
                       out_props=None, pass_ref=False):
        if in_props is None:
            in_props = {}
        if out_props is None:
            out_props = {}
        new_derivative = {
            'source': datastore,
            'func': derive_func,
            'in_props': in_props,
            'out_props': out_props,
            'pass_ref': pass_ref,
        }
        # check if exists and update
        for derivative in self._derivatives:
            if derivative['func'] == derive_func:
                derivative.update(new_derivative)
                break
        # or add new
        else:
            self._derivatives.append(new_derivative)

    def generate(self, key, return_val=False, **kwargs):
        """Generate data from the associated data generation method and
        store if metadata `save` is not set to False.

        Parameters
        ----------
        key : str
            Name of the data to generate.

        Returns
        -------
        Requested data after generating.

        """
        for (keys, generator) in self._generators.items():
            if key not in keys:
                continue

            datas = generator(**kwargs)
            if len(keys) == 1:
                if self.get_metadata(key, 'save', True):
                    self[key] = datas
                return datas if return_val else None

            if len(keys) != len(datas):
                raise ValueError(
                    'Data generator method did not return the expected '
                    'number of values.'
                )
            for key_i, data in zip(keys, datas):
                if self.get_metadata(key_i, 'save', True):
                    self[key_i] = data
                if key_i == key:
                    rtn_val = data
            return rtn_val if return_val else None

        else:
            raise DataGenerationError(f'Generator not found for data `{key}`')

    def update(self, other, priority=None):
        """Update with data items stored in `other`.

        Parameters
        ----------
        other : defdap.utils.Datastore
        priority : str
            Which datastore to keep an item from if the same name exists
            in both. Default is to prioritise `other`.

        """
        if priority == 'self':
            other._store.update(self._store)
            self._store = other._store
        else:
            self._store.update(other._store)

    def get_metadata(self, key, attr, value=None):
        """Get metadata value with a default returned if it does not
        exist. Imitating the `get()` method of a dictionary.

        Parameters
        ----------
        key : str
            Name of the data item.
        attr : str
            Metadata to get.
        value : any
            Default value to return if metadata does not exist.

        Returns
        -------
        Metadata value or the default value.

        """
        if key in self._store:
            return self._store[key].get(attr, value)

        try:
            return self._get_derived_item(key, attr)
        except KeyError:
            return value


class DataGenerationError(Exception):
    pass
