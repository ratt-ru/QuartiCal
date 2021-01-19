from collections import namedtuple
from quartical.kernels.complex import complex_solver
from quartical.kernels.phase import phase_solver
from quartical.kernels.kalman import kalman_solver


term_types = {"complex": namedtuple("cmplx", ("gains", "flags")),
              "phase": namedtuple("phase", ("gains", "flags", "params"))}

term_solvers = {"complex": complex_solver,
                "phase": phase_solver,
                "kalman": kalman_solver}

gain_templates = \
    {"complex": ("tipc", "fipc", "ant", "dir", "corr"),
     "phase": ("tipc", "fipc", "ant", "dir", "corr"),
     "delay": ("tpc", "fpc", "ant", "dir", "corr")}

param_templates = \
    {"complex": None,
     "phase": ("tipc", "fipc", "ant", "ppa", "dir", "corr"),
     "delay": ("tipc", "fipc", "ant", "ppa", "dir", "corr")}

n_ppa = \
    {"complex": None,
     "phase": 1,
     "delay": 2}

gain_spec_tup = namedtuple("gain_spec_tup",
                           "tchunk fchunk achunk dchunk cchunk")
param_spec_tup = namedtuple("param_spec_tup",
                            "tchunk fchunk achunk pchunk dchunk cchunk")


def make_chunk_specs(term_type, **kwargs):

    kwargs["ppa"] = n_ppa[term_type]  # Add the number of parameters per ant.
    kwargs = {k: v if isinstance(v, tuple) else (v,)
              for k, v in kwargs.items()}  # Convert to chunk-style tuple.

    gain_template = gain_templates[term_type]

    gain_spec = gain_spec_tup(*[kwargs.get(d) for d in gain_template])

    param_template = param_templates[term_type]

    if param_template:
        param_spec = param_spec_tup(*[kwargs.get(d) for d in param_template])
    else:
        param_spec = ()

    return gain_spec, param_spec
