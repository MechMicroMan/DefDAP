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

