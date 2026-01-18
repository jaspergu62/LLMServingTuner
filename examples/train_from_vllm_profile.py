#!/usr/bin/env python3
import argparse
from pathlib import Path

from llmservingtuner.inference.constant import OpAlgorithm
from llmservingtuner.model.xgb_state_model import StateXgbModel
from llmservingtuner.data_feature.dataset import CustomLabelEncoder, preset_category_data
try:
    from llmservingtuner.data_feature.dataset_with_modin import MyDataSetWithModin as MyDataSet
except ModuleNotFoundError:
    try:
        from llmservingtuner.data_feature.dataset_with_swifter import MyDataSetWithSwifter as MyDataSet
    except ModuleNotFoundError:
        from llmservingtuner.data_feature.dataset import MyDataSet

from llmservingtuner.train.pretrain import PretrainModel, TrainVersion1
from llmservingtuner.train.state_param import StateParam
from llmservingtuner.inference.utils import save_dataframe_to_csv


def _train_from_csv(feature_file: Path, output_dir: Path) -> None:
    train_files = [feature_file]

    sp = StateParam(
        base_path=output_dir,
        predict_field="model_execute_time",
        save_model=True,
        shuffle=True,
        plot_pred_and_real=False,
        plot_data_feature=False,
        plot_velocity_std=False,
        plot_predict_std=False,
        op_algorithm=OpAlgorithm.EXPECTED,
        xgb_model_show_test_data_prediction=False,
        xgb_model_show_feature_importance=False,
        plot_input_time_with_predict=False,
        title="MixModel without warmup with service info",
    )
    model = StateXgbModel(
        train_param=sp.xgb_model_train_param,
        update_param=sp.xgb_model_update_param,
        save_model_path=sp.xgb_model_save_model_path,
        load_model_path=sp.xgb_model_save_model_path,
        show_test_data_prediction=sp.xgb_model_show_test_data_prediction,
        show_feature_importance=sp.xgb_model_show_feature_importance,
    )

    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()

    dataset = MyDataSet(custom_encoder=custom_encoder, predict_field=sp.predict_field,
                        shuffle=sp.shuffle, op_algorithm=sp.op_algorithm)

    pm = PretrainModel(state_param=sp, dataset=dataset, model=model, plt_data=sp.plot_data_feature)
    TrainVersion1.simple_train(train_files, sp, pm)

    train_data = dataset.features.copy(deep=False)
    train_data["label"] = dataset.labels
    _train_dir = output_dir.joinpath("cache")
    save_dataframe_to_csv(train_data, _train_dir, "train_data.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simulation model from a vLLM profile CSV."
    )
    parser.add_argument("--profile-csv", required=True, type=Path, help="Path to the profile CSV")
    parser.add_argument("--output-dir", default=Path("model_output"), type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    profile_csv = args.profile_csv.expanduser().resolve()
    _train_from_csv(profile_csv, output_dir)


if __name__ == "__main__":
    main()
