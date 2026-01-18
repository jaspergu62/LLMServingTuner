#!/usr/bin/env python3
import argparse
import os
from argparse import Namespace
from pathlib import Path

from llmservingtuner.optimizer.optimizer import plugin_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run config tuning with the optimizer."
    )
    parser.add_argument("--engine", default="vllm", choices=["vllm", "mindie"])
    parser.add_argument("--benchmark", default="vllm_benchmark",
                        choices=["vllm_benchmark", "ais_bench"])
    parser.add_argument("--load-breakpoint", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--pd", default="competition", choices=["competition", "disaggregation"])
    parser.add_argument("--config", type=Path, help="Path to config.toml")
    args = parser.parse_args()

    if args.config:
        os.environ["LLMSERVINGTUNER_CONFIG_PATH"] = str(args.config.expanduser().resolve())

    plugin_args = Namespace(
        engine=args.engine,
        benchmark_policy=args.benchmark,
        load_breakpoint=args.load_breakpoint,
        backup=args.backup,
        pd=args.pd,
        deploy_policy="single",
    )
    plugin_main(plugin_args)


if __name__ == "__main__":
    main()
