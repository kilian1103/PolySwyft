class NRE_Settings:
    def __init__(self):
        """NRE initialisation.
        """
        self.root = "swyft_polychord_NSNRE"  # root directory
        self.child_root = "round"  # root for each round: {root}/{child_root}
        self.wandb_project_name = "NSNRE"
        self.neural_network_file = "NRE_network.pt"
        self.logger_name = f"{self.root}.log"
        self.seed = 234
        self.activate_wandb = False
        self.wandb_kwargs = {
            'project': self.wandb_project_name,
            'finish': {
                'exit_code': None,
                'quiet': None
            }
        }
        # simulator settings
        self.n_training_samples = 100_000  # nsamples for initial training using simulator
        self.n_weighted_samples = 10_000  # nsamples for evaluating NREs
        self.obsKey = "x"
        self.targetKey = "z"
        self.contourKey = "l"
        self.posteriorsKey = "post"
        self.num_features = 5
        self.num_features_dataset = 100
        self.num_mixture_components = 4
        # network settings
        self.num_summary_features = 5
        self.dropout = 0.3
        self.early_stopping_patience = 20
        self.learning_rate_init = 0.001
        self.learning_rate_decay = 0.95
        self.reset_learning_rate = True
        self.optimizer_file = "optimizer_file.pt"
        self.dm_kwargs = {
            'fractions': [0.8, 0.1, 0.1],
            'batch_size': 64,
            'shuffle': False,
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
        self.cyclic_rounds = True
        self.NRE_num_retrain_rounds = 10
        self.NRE_start_from_round = 0
        self.save_joint_training_data = True  # save joint training data for NRE retraining
        self.joint_training_data_fileroot = "joint_training_data.pt"
        self.termination_abs_dkl = 0.2
        self.n_DKL_estimates = 100
        self.use_noise_resampling = False
        self.n_noise_resampling_samples = 10
        self.use_dataset_truncation = False
        self.dataset_logR_cutoff = 3  # logR contour
        self.use_livepoint_increasing = False
        self.livepoint_increase_posterior_contour = 0.999  # zero point is at posterior peak
        self.n_increased_livepoints = 1_000
        self.increased_livepoints_fileroot = "enhanced_run"
        # plotting settings
        self.only_plot_mode = False
        self.plot_triangle_plot = True
        self.plot_triangle_plot_zoomed = True
        self.triangle_zoom_start = 8
        self.plot_KL_divergence = True
        self.plot_KL_compression = True
        self.plot_logR_histogram = True
        self.plot_logR_pdf = True
        self.plot_dataset_truncation = True
        self.plot_training_data = True
