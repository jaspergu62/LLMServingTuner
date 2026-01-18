#!/usr/bin/env python3
import argparse
from pathlib import Path

from llmservingtuner.train.source_to_train import source_to_model, req_decodetimes
from llmservingtuner.train.pretrain import pretrain


def _validate_profile_dir(profile_dir: Path, model_type: str) -> None:
    required = ["profiler.db", "request.csv"]
    if model_type == "vllm":
        required.append("kvcache.csv")
    missing = [name for name in required if not (profile_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing required files in {profile_dir}: {missing_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simulation model from profiling outputs."
    )
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("model_output"), type=Path)
    parser.add_argument("--type", choices=["vllm", "mindie"], default="vllm")
    args = parser.parse_args()

    profile_dir = args.profile_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    _validate_profile_dir(profile_dir, args.type)

    source_to_model(str(profile_dir), model_type=args.type)
    pretrain(profile_dir / "output_csv", output_dir)
    req_decodetimes(str(profile_dir), output_dir)


if __name__ == "__main__":
    main()
