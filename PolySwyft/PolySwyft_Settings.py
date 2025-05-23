class PolySwyft_Settings:
    def __init__(self, root):
        """NRE initialisation.
        """
        self.root = root
        # root directory
        self.child_root = "round"  # root for each round: {root}/{child_root}
        self.wandb_project_name = self.root
        self.neural_network_file = "NRE_network.pt"
        self.logger_name = f"{self.root}.log"
        self.seed = 234
        self.activate_wandb = True
        self.wandb_kwargs = {
            'project': self.wandb_project_name}
        # simulator settings
        self.n_training_samples = 10_000  # nsamples for initial training using simulator
        self.n_weighted_samples = 1_000  # nsamples for evaluating NREs
        self.obsKey = "x"
        self.targetKey = "z"
        self.contourKey = "l"
        self.posteriorsKey = "post"
        self.num_features = 5
        self.num_features_dataset = 100
        self.num_mixture_components = 4
        # network settings
        self.num_summary_features = 10
        self.dropout = 0.3
        self.early_stopping_patience = 20
        self.learning_rate_init = 0.001
        self.learning_rate_decay = 0.99
        self.optimizer_file = "optimizer_file.pt"
        self.dm_kwargs = {
            'fractions': [0.8, 0.1, 0.1],
            'batch_size': 64,
            'shuffle': False,
            'num_workers': 0
        }
        self.trainer_kwargs = {"accelerator": 'cpu',
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
        # polychord settings
        self.nderived = 0
        self.model = None
        # NSNRE settings
        self.continual_learning_mode = True
        self.cyclic_rounds = True # experimental, KL mode not working yet
        self.NRE_num_retrain_rounds = 20
        self.NRE_start_from_round = 0
        self.termination_abs_dkl = 0.2 #experimental, does not work properly yet
        self.n_DKL_estimates = 100
        self.use_noise_resampling = False
        self.n_noise_resampling_samples = 2
        self.use_livepoint_increasing = False
        self.livepoint_increase_posterior_contour = 0.999  # zero point is at posterior peak
        self.n_increased_livepoints = 1_000
        self.increased_livepoints_fileroot = "enhanced_run"
        # plotting settings
        self.only_plot_mode = False
        self.plot_triangle_plot = True
        self.triangle_start = 0
        self.plot_KL_divergence = True
        self.plot_statistical_power = True
