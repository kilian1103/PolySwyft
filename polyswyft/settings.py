"""Configuration container for the PolySwyft NSNRE cycle.

The :class:`PolySwyftSettings` class collects every knob exposed to the user
in a single mutable object that is passed through :class:`polyswyft.core.PolySwyft`,
:class:`polyswyft.dataloader.PolySwyftDataModule`, and the example simulators.

A typical pattern is::

    settings = PolySwyftSettings(root="MyExperiment")
    settings.NRE_num_retrain_rounds = 30
    settings.activate_wandb = False
    settings.dm_kwargs["batch_size"] = 128
"""


class PolySwyftSettings:
    """Container of all settings for a PolySwyft NSNRE run.

    All attributes are public and may be mutated after construction. Defaults
    target the toy examples shipped in ``examples/`` and should be reviewed
    before applying to a new problem.

    Parameters
    ----------
    root : str
        Top-level output directory for the run. All round artefacts
        (``{root}/{child_root}_{i}/``), the run log, and the pickled
        settings are placed here. Created on demand.

    Attributes
    ----------
    root : str
        Top-level output directory (also serves as default wandb project name).
    child_root : str
        Per-round subdirectory prefix; round ``i`` writes to
        ``{root}/{child_root}_{i}``. Default ``"round"``.
    neural_network_file : str
        Filename within each round directory for the saved NRE state-dict.
    optimizer_file : str
        Filename within each round directory for the saved optimiser state.
    logger_name : str
        Path of the run-level logfile (``"{root}.log"`` by default).
    seed : int
        Master random seed for ``pytorch_lightning.seed_everything`` and
        PolyChord.
    activate_wandb : bool
        If ``True``, attach a ``WandbLogger`` to the Lightning trainer for
        each round.
    wandb_project_name : str
        Wandb project, defaults to ``root``.
    wandb_kwargs : dict
        Keyword arguments forwarded to ``WandbLogger``.

    n_training_samples : int
        Number of prior samples used to seed the round-0 training set.
    n_weighted_samples : int
        Number of samples used when evaluating the trained NRE during plotting.
    obsKey : str
        Key under which the observed/simulated data ``D`` is stored in
        ``swyft.Sample`` dictionaries.
    targetKey : str
        Key for the parameter vector ``theta``.
    contourKey : str
        Key for true log-likelihood contours when generating analytical
        ground-truth samples.
    posteriorsKey : str
        Key for the analytical posterior samples (used in toy comparisons).
    num_features : int
        Dimensionality of ``theta``.
    num_features_dataset : int
        Dimensionality of ``D``.
    num_mixture_components : int
        Number of mixture components for the GMM example.

    num_summary_features : int
        Dimensionality of the optional data-compression summary statistic.
    dropout : float
        Dropout probability inside the swyft residual blocks.
    early_stopping_patience : int
        Patience (in epochs) for the Lightning ``EarlyStopping`` callback.
    learning_rate_init : float
        Initial learning rate for the Adam optimiser at round 0.
    learning_rate_decay : float
        Multiplicative decay applied per round by the user-supplied
        ``lr_round_scheduler`` (see ``examples/mvg/run_polyswyft.py``).

    dm_kwargs : dict
        Keyword arguments forwarded to :class:`PolySwyftDataModule`. Common
        keys: ``fractions`` (train/val/test split), ``batch_size``,
        ``shuffle``, ``num_workers``.
    trainer_kwargs : dict
        Keyword arguments forwarded to ``swyft.SwyftTrainer``
        (i.e. ``pytorch_lightning.Trainer``). Common keys: ``accelerator``,
        ``devices``, ``num_nodes``, ``strategy``, ``max_epochs``,
        ``precision``.

    nderived : int
        Number of derived parameters reported by PolyChord.
    model : Any
        Optional analytical model object (e.g. an ``lsbi`` model) used by the
        toy examples to compute the analytical posterior. Set by the example
        scripts after instantiating the simulator; ``None`` for real
        problems.

    continual_learning_mode : bool
        If ``True``, each round's NRE resumes from the previous round's
        weights; otherwise weights are re-initialised.
    cyclic_rounds : bool
        If ``True``, run a fixed number of rounds (``NRE_num_retrain_rounds``).
        If ``False``, terminate adaptively once
        ``KL(P_i || P_{i-1}) < termination_abs_dkl``.
        Note: the adaptive (``cyclic_rounds=False``) path is experimental.
    NRE_num_retrain_rounds : int
        Maximum number of NSNRE rounds.
    NRE_start_from_round : int
        Round index to resume from (uses on-disk artefacts of round
        ``NRE_start_from_round - 1``); set to ``0`` to start fresh.
    termination_abs_dkl : float
        KL threshold for the experimental adaptive-termination path.
    n_DKL_estimates : int
        Number of bootstrap repetitions used by ``anesthetic.logw`` /
        ``anesthetic.logZ`` to error-bar the KL diagnostics.

    use_noise_resampling : bool
        If ``True``, draw multiple ``D`` realisations per ``theta`` (Section
        3.6.3 of the paper) to enrich the training set.
    n_noise_resampling_samples : int
        Number of noise resamples per dead point when
        ``use_noise_resampling`` is on.
    use_livepoint_increasing : bool
        If ``True``, run a second PolyChord pass per round with extra live
        points injected at the contour selected by
        ``livepoint_increase_posterior_contour``.
    livepoint_increase_posterior_contour : float
        Cumulative posterior weight at which to boost live points (e.g.
        ``0.999`` targets the 99.9 % contour).
    n_increased_livepoints : int
        Live-point count to enforce at the boosted contour.
    increased_livepoints_fileroot : str
        Subdirectory name inside ``{root}/{child_root}_{i}/`` for the
        boosted run's chains.

    only_plot_mode : bool
        If ``True``, skip the NSNRE cycle and re-plot from existing artefacts
        on disk.
    plot_triangle_plot : bool
        Generate the per-round corner plots.
    triangle_start : int
        First round index to include in the triangle plot.
    plot_KL_divergence : bool
        Generate the KL divergence vs. round plot.
    plot_statistical_power : bool
        Generate the statistical-power diagnostic plot.
    """

    def __init__(self, root):
        self.root = root
        # Output dirs
        self.child_root = "round"
        self.neural_network_file = "NRE_network.pt"
        self.optimizer_file = "optimizer_file.pt"
        self.logger_name = f"{self.root}.log"

        # Reproducibility
        self.seed = 234

        # Wandb
        self.wandb_project_name = self.root
        self.activate_wandb = True
        self.wandb_kwargs = {"project": self.wandb_project_name}

        # Simulator / data shape
        self.n_training_samples = 10_000
        self.n_weighted_samples = 1_000
        self.obsKey = "x"
        self.targetKey = "z"
        self.contourKey = "l"
        self.posteriorsKey = "post"
        self.num_features = 5
        self.num_features_dataset = 100
        self.num_mixture_components = 4

        # Network architecture
        self.num_summary_features = 10
        self.dropout = 0.3
        self.early_stopping_patience = 20
        self.learning_rate_init = 0.001
        self.learning_rate_decay = 0.99

        # Lightning
        self.dm_kwargs = {"fractions": [0.8, 0.1, 0.1], "batch_size": 64, "shuffle": False, "num_workers": 0}
        self.trainer_kwargs = {
            "accelerator": "cpu",
            "devices": 60,
            "num_nodes": 1,
            "strategy": "ddp",
            "max_epochs": 1000,
            "log_every_n_steps": 1,
            "precision": 64,
            "enable_progress_bar": True,
            "default_root_dir": self.root,
            "callbacks": [],
            "deterministic": True,
        }

        # PolyChord
        self.nderived = 0
        self.model = None

        # NSNRE control
        self.continual_learning_mode = True
        self.cyclic_rounds = True
        self.NRE_num_retrain_rounds = 20
        self.NRE_start_from_round = 0
        self.termination_abs_dkl = 0.2
        self.n_DKL_estimates = 100

        # Dynamic nested sampling / data augmentation
        self.use_noise_resampling = False
        self.n_noise_resampling_samples = 2
        self.use_livepoint_increasing = False
        self.livepoint_increase_posterior_contour = 0.999
        self.n_increased_livepoints = 1_000
        self.increased_livepoints_fileroot = "enhanced_run"

        # Plotting
        self.only_plot_mode = False
        self.plot_triangle_plot = True
        self.triangle_start = 0
        self.plot_KL_divergence = True
        self.plot_statistical_power = True
