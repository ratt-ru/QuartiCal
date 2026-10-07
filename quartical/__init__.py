# -*- coding: utf-8 -*-
from quartical.data_handling.backend import activate_backend

# Route casacore imports to casacure before anything (in particular dask-ms)
# resolves them.
activate_backend()
