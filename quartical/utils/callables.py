import inspect


def filter_kwargs(f):
    """Wrap a function such that only necessary kwargs are passed in."""
    def wrapped(*args, **kwargs):
        sig = inspect.signature(f)
        filtered_kwargs = {k: kwargs[k] for k in sig.parameters.keys()}
        return f(*args, **filtered_kwargs)
    return wrapped
