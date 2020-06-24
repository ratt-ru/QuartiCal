from collections import defaultdict


def freeze_default_dict(ddict):
    """Freeze the contents of a nested defaultdict."""

    for k, v in ddict.items():
        if isinstance(v, defaultdict):
            ddict[k] = freeze_default_dict(v)

    return dict(ddict)
