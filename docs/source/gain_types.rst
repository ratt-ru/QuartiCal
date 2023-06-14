Gain Types
==========

This page details the various different gain types which QuartiCal can solve
for and includes some tips for ensuring you get good results.


Amplitude - ``amplitude``
-------------------------

This solves for amplitudes given gains of the following form (in the case of
linear feeds, but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} A^{XX} & 0 \\
                                 0 & A^{YY} \end{bmatrix}

where :math:`A` is a real-valued, positive amplitude.

Complex - ``complex``
---------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} g^{XX} & g^{XY} \\
                                 g^{YX} & g^{YY} \end{bmatrix}

.. note::
    * This term contains amplitude, phase and leakage information and may not
      be appropriate in the absence of a polarised model.

Crosshand Phase - ``crosshand_phase``
--------------------------------------

This solves for the crosshand phase given gains of the following form (in the
case of linear feeds, but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} e^{i\theta} & 0 \\
                                 0 & 1 \end{bmatrix}

where :math:`\theta` is the crosshand phase.

.. note::
    * This term requires four correlation data.
    * This term requires an excellent polarisation model.
    * This term must be solved over the entire array rather than per antenna.

Delay - ``delay``
-----------------

This solves for the delays given gains of the following form (in the case of
linear feeds, but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix}
        e^{i 2\pi d_{XX}(\nu - \nu_c)} & 0 \\
        0 & e^{i 2\pi d_{YY}(\nu - \nu_c)}
    \end{bmatrix}

where :math:`\nu` is the frequency of a particular channel, :math:`\nu_c` is
the central frequency and :math:`d` is the delay.

.. note::

    Delay solutions support the ``initial_estimate`` parameter. If specified,
    this will initialise the delays using the Fourier transform.

.. warning::

    Solving for a delay is very difficult if the phases are not approximately
    aligned. Thus, it is recommended to to solve for residual delay errors
    after applying a term which will approximately align the phases. This
    can be accomplished using e.g. ``solver.terms="[G,K]"`` where ``G`` is a 
    ``diag_complex`` term with long solution intervals.

Delay and Offset - ``delay_and_offset``
---------------------------------------

This solves for the delays and offsets (means) given gains of the following
form (in the case of linear feeds, but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix}
        e^{i (2\pi d_{XX}(\nu - \nu_c) + \theta_{XX})} & 0 \\
        0 & e^{i (2\pi d_{YY}(\nu - \nu_c) + \theta_{YY})}
    \end{bmatrix}

where :math:`\nu` is the frequency of a particular channel, :math:`\nu_c` is
the central frequency, :math:`d` is the delay and :math:`\theta` is some mean
phase offset.

.. note::

    Delay solutions support the ``initial_estimate`` parameter. If specified,
    this will initialise the delays using the Fourier transform.

.. warning::

    Solving for a delay is very difficult if the phases are not approximately
    aligned. Thus, it is recommended to to solve for residual delay errors
    after applying a term which will approximately align the phases. This
    can be accomplished using e.g. ``solver.terms="[G,K]"`` where ``G`` is a 
    ``diag_complex`` term with long solution intervals.

Diagonal Complex - ``diag_complex``
-----------------------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} g^{XX} & 0 \\
                                 0 & g^{YY} \end{bmatrix}

.. note::
    * This term contains amplitude and phase information but does not
      not incorporate leakage information. This makes it appropriate for
      the majority of use-cases.

Leakage - ``leakage``
---------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} 1 & d^{XY} \\
                                 d^{YX} & 1 \end{bmatrix}

where :math:`d` is a complex-valued quanitity which descibes the leakage.

Phase - ``phase``
-----------------

This solves for the phase given gains of the following form (in the case of
linear feeds, but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix} e^{i\theta_{XX}} & 0 \\
                                 0 & e^{i\theta_{YY}} \end{bmatrix}

Rotation - ``rotation``
-----------------------

This solves for gains of the following form (in the case of linear feeds,
but the same is true for circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix}
        \cos{\theta} & -\sin{\theta} \\
        \sin{\theta} & \cos{\theta}
    \end{bmatrix}

where :math:`\theta` is some unknown angle.

.. note::
    * This term is only applicable to four correlation data.
    * Solving for this term requires a polarised model.

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

.. note::
    * This term is only applicable to four correlation data.
    * Solving for this term requires a polarised model.

.. warning::

    This solver is highly experimental. Any problems should be reported via
    the issue tracker.

TEC and Offset - ``tec_and_offset``
-----------------------------------

This solves for the differential TEC values and offsets (means) given gains of
the following form (in the case of linear feeds, but the same is true for
circular feeds):

.. math::

    \mathbf{G} = \begin{bmatrix}
        e^{i (2\pi t_{XX}(\nu^{-1} + \frac{\log(\nu_{min}) - \log(\nu_{max})}{\nu_{max} - \nu_{min}}) + \theta_{XX})} & 0 \\
        0 & e^{i (2\pi t_{YY}(\nu^{-1} + \frac{\log(\nu_{min}) - \log(\nu_{max})}{\nu_{max} - \nu_{min}}) + \theta_{YY})} 
    \end{bmatrix}

where :math:`\nu` is the frequency of a particular channel, :math:`\nu_{min}`
is the smallest frequency, :math:`\nu_{max}` is the largest frequency,
:math:`t` is the differential (not absolute) TEC and :math:`\theta` is some
mean phase offset.

.. warning::

    This solver is highly experimental. Any problems should be reported via
    the issue tracker.


