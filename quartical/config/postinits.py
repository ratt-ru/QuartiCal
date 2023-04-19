from quartical.config.converters import as_time, as_freq


def __input_ms_post_init__(self):

    self.__validate_choices__()
    self.__validate_element_choices__()
    self.time_chunk = as_time(self.time_chunk)
    self.freq_chunk = as_freq(self.freq_chunk)

    assert len(self.select_uv_range) == 2, \
        "input_ms.select_uv_range expects a two-element list."

    assert not (self.sigma_column and self.weight_column), \
        "sigma_column and weight_column are mutually exclusive."

    if self.is_bda:
        assert self.time_chunk == 0, \
            ("input_ms.is_bda does not support chunking in time. Please "
                "set input_ms.time_chunk to 0.")
        assert self.freq_chunk == 0, \
            ("input_ms.is_bda does not support chunking in freq. Please "
                "set input_ms.freq_chunk to 0.")


def __input_model_post_init__(self):
    self.__validate_choices__()
    self.__validate_element_choices__()


def __output_post_init__(self):

    self.__validate_choices__()
    self.__validate_element_choices__()
    assert not (bool(self.products) ^ bool(self.columns)), \
        "Neither or both of products and columns must be specified."
    if self.products:
        assert len(self.products) == len(self.columns), \
                "Number of products not equal to number of columns."

    if self.net_gains:
        nested = any(isinstance(i, list) for i in self.net_gains)
        if nested:
            assert all(isinstance(i, list) for i in self.net_gains), \
                ("Contents of outputs.net_gains not understood. "
                    "Must be strictly a list or list of lists.")
        else:
            assert all(isinstance(i, str) for i in self.net_gains), \
                ("Contents of outputs.net_gains not understood. "
                    "Must be strictly a list or list of lists.")
            # In the non-nested case, introduce outer list (consistent).
            self.net_gains = [self.net_gains]


def __mad_flags_post_init__(self):
    self.__validate_choices__()
    self.__validate_element_choices__()


def __solver_post_init__(self):
    self.__validate_choices__()
    self.__validate_element_choices__()
    assert len(self.iter_recipe) >= len(self.terms), \
        "User has specified solver.iter_recipe with too few elements."

    assert self.convergence_criteria >= 1e-8, \
        "User has specified solver.convergence_criteria below 1e-8."


def __dask_post_init__(self):
    self.__validate_choices__()
    self.__validate_element_choices__()
    if self.address:
        msg = (
            "Scheduler address supplied but dask.scheduler has not "
            "been set to distributed."
        )
        assert self.scheduler == "distributed", msg


def __gain_post_init__(self):
    self.__validate_choices__()
    self.__validate_element_choices__()
    self.time_interval = as_time(self.time_interval)
    self.freq_interval = as_freq(self.freq_interval)
    if self.type == "crosshand_phase" and self.solve_per != "array":
        raise ValueError(
            "Crosshand phase can only be solved as a per array term. Please "
            "set the appropriate term.solve_per to 'array'."
        )


POST_INIT_MAP = {
    "input_ms": __input_ms_post_init__,
    "input_model": __input_model_post_init__,
    "output": __output_post_init__,
    "mad_flags": __mad_flags_post_init__,
    "solver": __solver_post_init__,
    "dask": __dask_post_init__,
    "gain": __gain_post_init__
}
