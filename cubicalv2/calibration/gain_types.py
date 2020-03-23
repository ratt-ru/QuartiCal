from collections import namedtuple


term_types = {"complex": namedtuple("cmplx", ("gains", "flags", "parms")),
              "phase": namedtuple("phase", ("gains", "flags", "parms"))}