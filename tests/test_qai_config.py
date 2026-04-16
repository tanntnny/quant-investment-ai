from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_prepare_qai_data_experiment_resolves_qai_prepare_flow() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=prepare_qai_data"])

    assert cfg.mode.name == "prepare_data"
    assert cfg.preparer._target_ == "src.preparers.qai_preparer.QaiPreparer"
    assert cfg.data.outputs.fundamental_filename == "fundamental.csv"
    assert cfg.data.outputs.story_filename == "story.csv"
    assert cfg.data.time_from == "20200101T0000"
    assert cfg.data.provider.alphavantage.divide_range_days == 90


def test_qai_preprocess_experiment_resolves_new_preprocess_flow() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=qai_preprocess"])

    assert cfg.mode.name == "prepare_data"
    assert cfg.preparer._target_ == "src.preparers.qai_preprocess.QaiPreprocessPreparer"
    assert cfg.data.inputs.fundamental_path == "data/cleaned/fundamental.csv"
    assert cfg.data.outputs.preprocessed_filename == "qai_preprocessed.csv"


def test_qai_train_experiment_resolves_train_flow() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=qai_train"])

    assert cfg.mode.name == "train"
    assert cfg.data._target_ == "src.datamodules.qai_datamodule.QaiDataModule"
    assert cfg.model._target_ == "src.models.qai_attention.QaiAttentionModel"
    assert cfg.loss._target_ == "src.losses.qai_multitask.QaiMultiTaskLoss"
    assert cfg.metrics._target_ == "src.metrics.qai_multitask.QaiMultiTaskMetrics"
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.trainer._target_ == "src.trainers.pytorch_trainer.PytorchTrainer"


def test_qai_portfolio_model_pair_configs_resolve() -> None:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        bilstm_cfg = compose(
            config_name="config",
            overrides=["experiment=qai_portfolio_train", "portfolio_model=bilstm"],
        )
        lightgbm_cfg = compose(
            config_name="config",
            overrides=["experiment=qai_portfolio_train", "portfolio_model=lightgbm"],
        )

    assert bilstm_cfg.model._target_ == "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel"
    assert bilstm_cfg.trainer._target_ == "src.trainers.pytorch_trainer.PytorchTrainer"
    assert (
        lightgbm_cfg.model._target_
        == "src.models.qai_portfolio_lightgbm.QaiPortfolioLightGBMModel"
    )
    assert (
        lightgbm_cfg.trainer._target_
        == "src.trainers.lightgbm_portfolio_trainer.LightGBMPortfolioTrainer"
    )
