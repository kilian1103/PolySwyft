Quickstart
==========

The shortest end-to-end PolySwyft run uses the multivariate Gaussian
example shipped under ``examples/mvg/``. The script below reproduces
section 4.1 of the paper.

.. code-block:: bash

    pip install "polyswyft[examples,mpi]"
    mpirun -n 60 python -m examples.mvg.run_polyswyft

Skeleton
--------

A minimal driver consists of three pieces: a ``swyft.Simulator``, a
:class:`~polyswyft.network.PolySwyftNetwork` subclass, and a
``pypolychord.PolyChordSettings``.

.. code-block:: python

    from polyswyft import PolySwyft, PolySwyftSettings
    import pypolychord, swyft

    settings = PolySwyftSettings(root="MyRun")
    settings.NRE_num_retrain_rounds = 14

    sim = MySimulator(settings)              # subclass of swyft.Simulator
    obs = swyft.Sample(x=...)                # observed data D_obs
    network = MyNetwork(settings, obs)       # subclass of PolySwyftNetwork

    polyset = pypolychord.PolyChordSettings(settings.num_features, nDerived=0)
    polyset.nlive = 100 * settings.num_features

    PolySwyft(
        polyswyftSettings=settings,
        sim=sim,
        obs=obs,
        deadpoints=initial_prior_samples,    # (N, num_features)
        network=network,
        polyset=polyset,
        callbacks=lambda: [],
    ).execute_NSNRE_cycle()

The full reference is in :mod:`polyswyft.core`. For a complete worked
example see ``examples/mvg/run_polyswyft.py`` in the repository.
