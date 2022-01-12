Gain Types
==========

This page details the various different gain types which QuartiCal can solve
for and includes some tips for ensuring you get good results.

Complex - ``complex``
---------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} g^{XX} & g^{XY} \\ 
                                 g^{YX} & g^{YY} \end{bmatrix} 

Note:
    * This term contains amplitude, phase and leakage information and may not
      be appropriate in the absence of a polarised model.
    * Can be used in conjunction with ``input_ms.select_corr`` to solve over
      a subset of the correlations present in the measurement set.


Diagonal Complex - ``diag_complex``
-----------------------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} g^{XX} & 0 \\ 
                                 0 & g^{YY} \end{bmatrix} 

Note:
    * This term contains amplitude and phase information but does not
      not incorporate leakage information. This makes it approporiate for
      the majority of use-cases.


Approximate Complex - ``approx_complex``
----------------------------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} g^{XX} & g^{XY} \\ 
                                 g^{YX} & g^{YY} \end{bmatrix} 

Note:
    * This term contains amplitude, phase and leakage information and may not
      be appropriate in the absence of a polarised model.
    * This is roughly equivalent to the ``complex`` case but makes some
      simplifying assumptions about the data weights. This makes it faster
      but less accurate.
    * Can be used in conjunction with ``input_ms.select_corr`` to solve over
      a subset of the correlations present in the measurement set.

.. warning::

    This solver may be deprecated at short notice if it causes problems for
    users.

Amplitude - ``amplitude``
-------------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} |g^{XX}| & 0 \\ 
                                 0 & |g^{YY}| \end{bmatrix} 

Note:
    * This term contains only amplitude information and should be used with 
      caution. Amplitude only solutions are prone to introducing ghosts in the
      presence of an incomplete model.


Phase - ``phase``
-----------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} e^{\theta_{XX}} & 0 \\ 
                                 0 & e^{\theta_{YY}} \end{bmatrix} 

Note:
    * This term contains only phase information and should be used with 
      caution. Phase only solutions are prone to introducing ghosts in the
      presence of an incomplete model.


Delay - ``delay``
-----------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} \exp(a_{XX}\nu + b_{XX}) & 0 \\ 
                                 0 & \exp(a_{YY}\nu + b_{YY}) \end{bmatrix} 

where :math:`\nu` is the frequency of a particular channel, :math:`a` is the
delay and :math:`b` is a phase offset.

Note:
    * This term contains only phase information and should be used with 
      caution. Phase only solutions are prone to introducing ghosts in the
      presence of an incomplete model.

.. warning::

    Solving for a delay is very difficult if the phases are not approximately
    aligned. Thus, it is recommeded to to solve for residual delay errors 
    after applying a term which will approximately align the phases. This 
    can be accomplished using e.g. ``solver.terms=[K,G,B]`` and 
    ``solver.iter_recipe=[0,25,25,25,25,25]``. This means that the delay is
    only solved after first letting the gain and bandpass solutions
    approximately align the phases.


TEC - ``tec``
-------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} 
        \exp(a_{XX}{\nu}^{-1} + b_{XX}) & 0 \\ 
        0 & \exp(a_{YY}{\nu}^{-1} + b_{YY})
    \end{bmatrix} 

where :math:`\nu` is the frequency of a particular channel, :math:`a` is the
tec multiplied by some constants and :math:`b` is a phase offset.

Note:
    * This term contains only phase information and should be used with 
      caution. Phase only solutions are prone to introducing ghosts in the
      presence of an incomplete model.

.. warning::

    This solver is highly experimental. Any problems should be reported via
    the issue tracker.


Rotation Measure - ``rotation_measure``
---------------------------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix}
        \cos{(\lambda^2\mathrm{RM})} & -\sin{(\lambda^2\mathrm{RM})} \\
        \sin{(\lambda^2\mathrm{RM})} & \cos{(\lambda^2\mathrm{RM})}
    \end{bmatrix}

where :math:`\lambda` is the wavelength in a particular channel and 
:math:`\mathrm{RM}` is an estimate of the rotation measure.

Note:
    * This terms is only applicable to four correlation data.
    * Solving for this term requires a polarised model.

.. warning::

    This solver is highly experimental. Any problems should be reported via
    the issue tracker.