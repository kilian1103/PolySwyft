Algorithm
=========

PolySwyft executes a five-phase NSNRE cycle (see paper Figure 1).

1. **Initialise.** Draw ``N^(0)`` samples from the prior to seed the
   round-0 training set.
2. **Simulate.** Forward-simulate ``D`` for each parameter sample to
   form joint pairs ``(theta, D)``. Disjoints are constructed
   internally by swyft via within-batch shuffling.
3. **Train NRE.** Concatenate the round's new samples with the
   cumulative training set and fit the binary classifier
   :math:`p(M_J | \theta, D)`. Under
   ``polyswyftSettings.continual_learning_mode`` the network resumes
   from the previous round's checkpoint.
4. **Run NS.** Invoke PolyChord with ``network.logRatio`` as the
   likelihood callback to draw new dead points around ``D_obs``.
5. **Terminate.** Compute :math:`\mathrm{KL}(P_i \| P_{i-1})` and
   :math:`\mathrm{KL}(P_i \| \pi)`. The current implementation supports
   two termination paths, controlled by
   ``polyswyftSettings.cyclic_rounds``:

   * ``cyclic_rounds=True`` (default) — run a fixed number of rounds
     (``NRE_num_retrain_rounds``). Both KL diagnostics are still
     reported per round; the practitioner inspects them post-hoc to
     judge whether enough compression was achieved (paper section 3.4
     refers to this manual check as the ``C_comp`` criterion).
   * ``cyclic_rounds=False`` (experimental) — stop once
     ``|KL(P_i || P_{i-1})| < termination_abs_dkl``. There is no
     automatic ``C_comp`` enforcement; the user is expected to choose
     ``termination_abs_dkl`` conservatively and watch the compression
     plot.

The cycle is implemented in :class:`polyswyft.core.PolySwyft`. KL
diagnostics live in :mod:`polyswyft.utils`.

Key equations
-------------

.. math::

    r(\theta, D) = \frac{p(D|\theta)}{p(D)} = \frac{\mathcal{P}}{\pi}
                 = \frac{r^*(\theta, D)}{Z_{NS}}

where :math:`Z_{NS}` is the per-round normalisation that accounts for
training the NRE on a dead-measure distribution rather than directly on
the prior. See paper section 3.4 for the full derivation.

For the convergence criterion (paper eq. 10):

.. math::

    \mathrm{KL}(\mathcal{P}_i \| \mathcal{P}_{i-1})
    \approx \sum_{n=1}^{N} w_n \big[
        \log r_i(\theta_n, D_{obs}) - \log Z_{NS,i}
        - \log r_{i-1}(\theta_n, D_{obs}) + \log Z_{NS,i-1}
    \big]

with :math:`\theta_n \sim p_i(\theta | D_{obs})` and
:math:`\sum_n w_n = 1`.
