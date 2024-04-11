from collections.abc import MutableMapping
from collections import defaultdict


def freeze_default_dict(ddict):
    """Freeze the contents of a nested defaultdict."""

    for k, v in ddict.items():
        if isinstance(v, defaultdict):
            ddict[k] = freeze_default_dict(v)

    return dict(ddict)


def flatten(dictionary, parent_key=()):
    """Flatten dictionary. Adapted from https://stackoverflow.com/a/6027615."""
    items = []

    for key, value in dictionary.items():
        new_key = parent_key + (key,) if parent_key else (key,)
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key).items())
        else:
            items.append((new_key, value))

    return dict(items)
