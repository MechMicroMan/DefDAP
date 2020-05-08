import functools

def reportProgress(message=""):
    """Decorator for reporting progress of given function

    Parameters
    ----------
    message : str
        Message to display (prefixed by 'Starting ', progress percentage and then 'Finished '

    References
    ----------
    Inspiration from :
    https://gist.github.com/Garfounkel/20aa1f06234e1eedd419efe93137c004

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            messageStart = "\rStarting " + message + ".."
            print(messageStart, end="")
            # The yield statements in the function produces a generator
            generator = func(*args, **kwargs)
            progPrev = 0.
            printFinal = True
            try:
                while True:
                    prog = next(generator)
                    if type(prog) is str:
                        printFinal = False
                        print("\r" + prog)
                        continue
                    # only report each percent
                    if prog - progPrev > 0.01:
                        messageProg = messageStart + \
                                      " {:} % ".format(int(prog*100))
                        print(messageProg, end="")
                        progPrev = prog
                        printFinal = True

            except StopIteration as e:
                if printFinal:
                    messageEnd = "\rFinished " + message + "           "
                    print(messageEnd)
                # When generator finished pass the return value out
                return e.value

        return wrapper
    return decorator

