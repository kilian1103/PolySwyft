"""Tests for PolySwyft.PolySwyftSettings.PolySwyftSettings."""

from polyswyft.settings import PolySwyftSettings


def test_settings_default_root():
    settings = PolySwyftSettings(root="my_root")
    assert settings.root == "my_root"


def test_settings_child_root_default():
    settings = PolySwyftSettings(root="r")
    assert settings.child_root == "round"


def test_settings_file_defaults():
    settings = PolySwyftSettings(root="r")
    assert settings.neural_network_file == "NRE_network.pt"
    assert settings.optimizer_file == "optimizer_file.pt"
    assert settings.logger_name == "r.log"


def test_settings_default_numeric_params():
    settings = PolySwyftSettings(root="r")
    assert settings.num_features == 5
    assert settings.num_features_dataset == 100
    assert settings.num_mixture_components == 4
    assert settings.n_training_samples == 10_000
    assert settings.n_weighted_samples == 1_000
    assert settings.num_summary_features == 10
    assert settings.dropout == 0.3
    assert settings.early_stopping_patience == 20
    assert settings.learning_rate_init == 0.001
    assert settings.learning_rate_decay == 0.99
    assert settings.n_DKL_estimates == 100
    assert settings.seed == 234


def test_settings_sample_keys_defaults():
    settings = PolySwyftSettings(root="r")
    assert settings.obsKey == "x"
    assert settings.targetKey == "z"
    assert settings.contourKey == "l"
    assert settings.posteriorsKey == "post"


def test_settings_dm_kwargs_structure():
    settings = PolySwyftSettings(root="r")
    assert set(settings.dm_kwargs.keys()) == {"fractions", "batch_size", "shuffle", "num_workers"}
    assert settings.dm_kwargs["fractions"] == [0.8, 0.1, 0.1]
    assert settings.dm_kwargs["batch_size"] == 64
    assert settings.dm_kwargs["shuffle"] is False
    assert settings.dm_kwargs["num_workers"] == 0


def test_settings_trainer_kwargs_structure():
    settings = PolySwyftSettings(root="r")
    expected_keys = {
        "accelerator",
        "devices",
        "num_nodes",
        "strategy",
        "max_epochs",
        "log_every_n_steps",
        "precision",
        "enable_progress_bar",
        "default_root_dir",
        "callbacks",
        "deterministic",
    }
    assert expected_keys.issubset(set(settings.trainer_kwargs.keys()))
    assert settings.trainer_kwargs["max_epochs"] == 1000
    assert settings.trainer_kwargs["precision"] == 64
    assert settings.trainer_kwargs["deterministic"] is True
    assert settings.trainer_kwargs["default_root_dir"] == settings.root


def test_settings_wandb_project_name_matches_root():
    settings = PolySwyftSettings(root="my_project")
    assert settings.wandb_project_name == "my_project"
    assert settings.wandb_kwargs["project"] == "my_project"


def test_settings_model_initially_none():
    settings = PolySwyftSettings(root="r")
    assert settings.model is None
    assert settings.nderived == 0


def test_settings_boolean_defaults():
    settings = PolySwyftSettings(root="r")
    assert settings.continual_learning_mode is True
    assert settings.cyclic_rounds is True
    assert settings.activate_wandb is True
    assert settings.use_noise_resampling is False
    assert settings.use_livepoint_increasing is False
    assert settings.plot_triangle_plot is True
    assert settings.plot_KL_divergence is True
    assert settings.plot_statistical_power is True
    assert settings.only_plot_mode is False


def test_settings_rounds_defaults():
    settings = PolySwyftSettings(root="r")
    assert settings.NRE_num_retrain_rounds == 20
    assert settings.NRE_start_from_round == 0
    assert settings.termination_abs_dkl == 0.2


def test_settings_livepoint_defaults():
    settings = PolySwyftSettings(root="r")
    assert settings.livepoint_increase_posterior_contour == 0.999
    assert settings.n_increased_livepoints == 1_000
    assert settings.increased_livepoints_fileroot == "enhanced_run"
    assert settings.n_noise_resampling_samples == 2


def test_settings_mutability():
    settings = PolySwyftSettings(root="r")
    settings.num_features = 13
    settings.dropout = 0.5
    assert settings.num_features == 13
    assert settings.dropout == 0.5


def test_settings_independent_instances():
    """Two instances should not share state."""
    a = PolySwyftSettings(root="a")
    b = PolySwyftSettings(root="b")
    a.num_features = 99
    assert b.num_features == 5  # unchanged
